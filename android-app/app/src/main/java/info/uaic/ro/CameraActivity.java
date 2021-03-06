package info.uaic.ro;

import android.Manifest;
import android.app.Activity;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.*;
import android.util.Size;
import android.view.Surface;
import android.view.WindowManager;
import android.widget.Toast;
import info.uaic.ro.common.OverlayView;
import info.uaic.ro.env.ImageUtils;
import info.uaic.ro.env.Logger;
import info.uaic.ro.surface_crack_detection.R;

import java.nio.ByteBuffer;

public abstract class CameraActivity extends Activity
        implements OnImageAvailableListener, Camera.PreviewCallback
{
    private static final Logger LOGGER = new Logger();

    private static final int PERMISSIONS_REQUEST = 1;

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private final byte[][] yuvBytes = new byte[3][];
    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private Handler handler;
    private HandlerThread handlerThread;
    private boolean useCamera2API;
    private boolean isProcessingFrame = false;
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;

    @Override
    protected void onCreate(final Bundle savedInstanceState)
    {
        LOGGER.d("onCreate " + this);
        super.onCreate(null);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);

        if (hasPermission())
        {
            setFragment();
        }
        else
        {
            requestPermission();
        }
    }

    protected int[] getRgbBytes()
    {
        imageConverter.run();
        return rgbBytes;
    }

    /**
     * Callback for android.hardware.Camera API
     */
    @Override
    public void onPreviewFrame(final byte[] bytes, final Camera camera)
    {
        if (isProcessingFrame)
        {
            LOGGER.w("Dropping frame!");
            return;
        }

        try
        {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null)
            {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewWidth = previewSize.width;
                previewHeight = previewSize.height;

                LOGGER.i("Preview Size (%d, %d), Orientation %d", previewWidth, previewHeight, 90);

                rgbBytes = new int[previewWidth * previewHeight];
                onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
            }
        }
        catch (final Exception e)
        {
            LOGGER.e(e, "Exception!");
            return;
        }

        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        yRowStride = previewWidth;

        imageConverter = () -> ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);

        postInferenceCallback = () ->
        {
            camera.addCallbackBuffer(bytes);
            isProcessingFrame = false;
        };
        processImage();
    }

    /**
     * Callback for Camera2 API
     */
    @Override
    public void onImageAvailable(final ImageReader reader)
    {
        //We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0)
        {
            return;
        }
        if (rgbBytes == null)
        {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try
        {
            final Image image = reader.acquireLatestImage();

            if (image == null)
            {
                return;
            }

            if (isProcessingFrame)
            {
                image.close();
                return;
            }
            isProcessingFrame = true;

            Trace.beginSection("imageAvailable");
            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter = () -> ImageUtils.convertYUV420ToARGB8888(yuvBytes[0], yuvBytes[1], yuvBytes[2], previewWidth, previewHeight,
                    yRowStride, uvRowStride, uvPixelStride, rgbBytes);

            postInferenceCallback = () ->
            {
                image.close();
                isProcessingFrame = false;
            };

            processImage();
        }
        catch (final Exception e)
        {
            LOGGER.e(e, "Exception!");

            Trace.endSection();
            return;
        }
        Trace.endSection();
    }

    @Override
    public synchronized void onStart()
    {
        LOGGER.d("onStart " + this);
        super.onStart();
    }

    @Override
    public synchronized void onResume()
    {
        LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("front");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public synchronized void onPause()
    {
        LOGGER.d("onPause " + this);

        if (!isFinishing())
        {
            LOGGER.d("Requesting finish");
            finish();
        }

        handlerThread.quitSafely();
        try
        {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        }
        catch (final InterruptedException e)
        {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    @Override
    public synchronized void onStop()
    {
        LOGGER.d("onStop " + this);
        super.onStop();
    }

    @Override
    public synchronized void onDestroy()
    {
        LOGGER.d("onDestroy " + this);
        super.onDestroy();
    }

    protected synchronized void runInBackground(final Runnable r)
    {
        if (handler != null)
        {
            handler.post(r);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults)
    {
        if (requestCode == PERMISSIONS_REQUEST)
        {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED)
            {
                setFragment();
            }
            else
            {
                requestPermission();
            }
        }
    }

    private boolean hasPermission()
    {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
        {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        }
        else
        {
            return true;
        }
    }

    private void requestPermission()
    {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
        {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
                    shouldShowRequestPermissionRationale(PERMISSION_STORAGE))
            {
                Toast.makeText(CameraActivity.this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[]{PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private boolean isHardwareLevelSupported(
            CameraCharacteristics characteristics)
    {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY)
        {
            return false;
        }
        // deviceLevel is not LEGACY, can use numerical sort
        return android.hardware.camera2.CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL <= deviceLevel;
    }

    private String chooseCamera()
    {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try
        {
            for (final String cameraId : manager.getCameraIdList())
            {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null || facing == CameraCharacteristics.LENS_FACING_FRONT)
                {
                    continue;
                }

                final StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null)
                {
                    continue;
                }

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                        || isHardwareLevelSupported(characteristics
                );
                LOGGER.i("Camera API lv2: %s", useCamera2API);
                return cameraId;
            }
        }
        catch (CameraAccessException e)
        {
            LOGGER.e(e, "Not allowed to access camera");
        }

        return null;
    }

    protected void setFragment()
    {
        String cameraId = chooseCamera();

        Fragment fragment;
        if (useCamera2API)
        {
            LOGGER.d("Use Camera 2 API");
            CameraConnectionFragment camera2Fragment = CameraConnectionFragment.newInstance(
            (size, rotation) ->
            {
                previewHeight = size.getHeight();
                previewWidth = size.getWidth();
                CameraActivity.this.onPreviewSizeChosen(size, rotation);
            },
            this,
            getLayoutId(),
            getDesiredPreviewFrameSize());
            camera2Fragment.setCamera(cameraId);
            fragment = camera2Fragment;
        }
        else
        {
            fragment = new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
        }

        getFragmentManager()
                .beginTransaction()
                .replace(R.id.container, fragment)
                .commit();
    }

    protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes)
    {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i)
        {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null)
            {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    public void requestRender()
    {
        final OverlayView overlay = findViewById(R.id.debug_overlay);
        if (overlay != null)
        {
            overlay.postInvalidate();
        }
    }

    public void addCallback(final OverlayView.DrawCallback callback)
    {
        final OverlayView overlay = findViewById(R.id.debug_overlay);
        if (overlay != null)
        {
            overlay.addCallback(callback);
        }
    }

    protected void readyForNextImage()
    {
        if (postInferenceCallback != null)
        {
            postInferenceCallback.run();
        }
    }

    protected int getScreenOrientation()
    {
        switch (getWindowManager().getDefaultDisplay().getRotation())
        {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    protected abstract void processImage();

    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

    protected abstract int getLayoutId();

    protected abstract Size getDesiredPreviewFrameSize();
}
