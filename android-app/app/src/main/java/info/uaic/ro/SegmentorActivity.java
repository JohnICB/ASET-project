package info.uaic.ro;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;
import info.uaic.ro.common.BorderedText;
import info.uaic.ro.common.OverlayView;
import info.uaic.ro.env.ImageUtils;
import info.uaic.ro.env.Logger;
import info.uaic.ro.segmentation.Segmentor;
import info.uaic.ro.segmentation.TFLiteObjectSegmentationAPIModel;
import info.uaic.ro.surface_crack_detection.R;
import info.uaic.ro.tracking.OverlayTracker;

import java.io.IOException;
import java.util.Vector;

/**
 * An activity that uses a TensorFlowMultiBoxsegmentor and ObjectTracker to segment and then track
 * objects.
 */
public class SegmentorActivity extends CameraActivity implements OnImageAvailableListener
{
    private static final Logger LOGGER = new Logger();
    // Configuration values for the prepackaged DeepLab model.
    private static final int TF_OD_API_INPUT_WIDTH = 256;
    private static final int TF_OD_API_INPUT_HEIGHT = 256;
    private static final int TF_OD_API_NUM_CLASS = 2;
    private static final String TF_OD_API_MODEL_FILE = "model_float16.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labels.txt";
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    static
    {
        System.loadLibrary("opencv_java");
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
    }

    OverlayView segmentOverlay;

    private Segmentor segmentor;

    private long lastProcessingTimeMs;
    private long lastInferenceTimeMs;
    private long lastNativeTimeMs;
    private Vector<String> lastLabels;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    private boolean computingSegmentation = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;

    private OverlayTracker tracker;

    private BorderedText borderedText;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation)
    {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        try
        {
            segmentor =
                    TFLiteObjectSegmentationAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_WIDTH,
                            TF_OD_API_INPUT_HEIGHT,
                            TF_OD_API_NUM_CLASS
                    );
            ImageView overlay = findViewById(R.id.overlay);
            overlay.setVisibility(View.VISIBLE);

            tracker = new OverlayTracker(DESIRED_PREVIEW_SIZE, segmentor.getLabels());
        }
        catch (final IOException e)
        {
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast = Toast.makeText(
                    getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        int cropHeight = TF_OD_API_INPUT_HEIGHT;
        int cropWidth = TF_OD_API_INPUT_WIDTH;

        int sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropWidth, cropHeight,
                sensorOrientation, MAINTAIN_ASPECT);

        Matrix cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        segmentOverlay = findViewById(R.id.segment_overlay);
        segmentOverlay.addCallback(canvas -> tracker.draw(canvas));

        addCallback(canvas ->
        {
            if (lastLabels == null)
            {
                return;
            }

            final Vector<String> lines = new Vector<>();
            lines.add("Project:");
            lines.add("- Title: Real-Time Surface crack detection");
            lines.add("");
            lines.add("Info:");
            lines.add("- TF Lite(Native) Time: " + lastNativeTimeMs + "ms");
            lines.add("- TF Lite(Java) Overhead: " + (lastInferenceTimeMs - lastNativeTimeMs) + "ms");
            lines.add("- Pre/Post Processing Overhead: " + (lastProcessingTimeMs - lastInferenceTimeMs) + "ms");
            lines.add("- Labels: " + String.join(", ", lastLabels));
            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
        });
    }

    @Override
    protected void processImage()
    {
        ++timestamp;
        final long currTimestamp = timestamp;
        segmentOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingSegmentation)
        {
            readyForNextImage();
            return;
        }
        computingSegmentation = true;
        LOGGER.i("Preparing image " + currTimestamp + " for segmention in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP)
        {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(() ->
        {
            LOGGER.i("Running segmention on image " + currTimestamp);

            final long startTime = SystemClock.uptimeMillis();
            final Segmentor.Segmentation result = segmentor.segmentImage(croppedBitmap);
            lastInferenceTimeMs = result.getInferenceTime();
            lastNativeTimeMs = result.getNativeTime();
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            tracker.trackResults(result, currTimestamp);
            lastLabels = tracker.getLastLabels();

            segmentOverlay.postInvalidate();
            requestRender();
            computingSegmentation = false;
        });
    }

    @Override
    protected int getLayoutId()
    {
        return R.layout.camera_connection_fragment_segment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize()
    {
        return DESIRED_PREVIEW_SIZE;
    }
}
