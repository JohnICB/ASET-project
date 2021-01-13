package info.uaic.ro.segmentation;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import info.uaic.ro.env.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.Vector;

public class TFLiteObjectSegmentationAPIModel implements Segmentor
{
    private static final Logger LOGGER = new Logger();
    public Vector<String> labels = new Vector<>();
    protected FloatBuffer imgData = null;
    // Config values.
    private int inputWidth;
    private int inputHeight;
    private int numClass;
    // Pre-allocated buffers.
    private int[] intValues;
    private float[][][][] pixelClasses;
    private Interpreter tfLite;

    private TFLiteObjectSegmentationAPIModel()
    {
    }

    /**
     * Memory-map the model file in Assets.
     */
    private static ByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException
    {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     */
    public static Segmentor create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputWidth,
            final int inputHeight,
            final int numClass) throws IOException
    {
        final TFLiteObjectSegmentationAPIModel d = new TFLiteObjectSegmentationAPIModel();

        d.inputWidth = inputWidth;
        d.inputHeight = inputHeight;
        d.numClass = numClass;

        try
        {
            Interpreter.Options options = new Interpreter.Options();
            options.setUseXNNPACK(true);
            options.setNumThreads(8);
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }

        InputStream labelsInput;
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        labelsInput = assetManager.open(actualFilename);
        BufferedReader br;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null)
        {
            d.labels.add(line);
        }

        // Pre-allocate buffers.
        d.imgData = FloatBuffer.allocate(d.inputWidth * d.inputHeight * 3);
        d.intValues = new int[d.inputWidth * d.inputHeight];
        d.pixelClasses = new float[1][d.inputHeight][d.inputWidth][1];
        return d;
    }

    public Vector<String> getLabels()
    {
        return labels;
    }

    public Segmentation segmentImage(final Bitmap bitmap)
    {
        if (imgData != null)
        {
            imgData.rewind();
        }

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("segmentImage");
        Trace.beginSection("preprocessBitmap");
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int j = 0; j < inputHeight; ++j)
        {
            for (int i = 0; i < inputWidth; ++i)
            {
                int pixel = intValues[j * inputWidth + i];
                imgData.put((float) ((pixel >> 16) & 0xFF));
                imgData.put((float) ((pixel >> 8) & 0xFF));
                imgData.put((float) (pixel & 0xFF));
            }
        }
        Trace.endSection(); // preprocessBitmap

        // Run the inference call.
        Trace.beginSection("run");
        long startTime = SystemClock.uptimeMillis();
        tfLite.run(prepareInput(imgData), pixelClasses);
        long endTime = SystemClock.uptimeMillis();
        Trace.endSection(); // run

        Trace.endSection(); // segmentImage

        return new Segmentation(prepareOutput(), numClass, inputWidth, inputHeight, endTime - startTime,
                tfLite.getLastNativeInferenceDurationNanoseconds() / 1000 / 1000);
    }

    private float[][][][] prepareInput(FloatBuffer bytes)
    {
        bytes.rewind();
        bytes.put(new ClipByValue(new NDArray(bytes.array(), new int[]{1, bytes.capacity()}, 'c'),
                0.0f, 255.0f).getInputArgument(0).div(255.0f).toFloatVector());

        bytes.rewind();
        float[][][][] prepared = new float[1][256][256][3];
        for (int i = 0; i < this.inputWidth; i++)
        {
            for (int j = 0; j < this.inputHeight; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    int index = i * this.inputWidth * 3 + j * 3 + k;
                    prepared[0][i][j][k] = (bytes.get(index));
                }
            }
        }

        return prepared;
    }

    private float[] prepareOutput()
    {
        INDArray array = new ClipByValue(new NDArray(ArrayUtil.flatten(this.pixelClasses), new int[]{1, this.inputWidth * this.inputHeight}, 'c'), 0, 1).getInputArgument(0).mul(255.0f);
        float mean = array.mean(0).getFloat();

        Mat resMat = new Mat(this.inputHeight, this.inputWidth, CvType.CV_32FC1);
        resMat.put(0, 0, array.toFloatVector());
        Imgproc.threshold(resMat, resMat, mean, 255, Imgproc.THRESH_BINARY);
        float[] output = new float[(int) resMat.total()];
        resMat.get(0, 0, output);
        return output;
    }

}
