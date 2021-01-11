package info.uaic.ro.tracking;

import android.graphics.*;
import android.util.Size;
import info.uaic.ro.env.Logger;
import info.uaic.ro.segmentation.Segmentor;

import java.util.Vector;

/**
 * A tracker wrapping ObjectTracker that also handles non-max suppression and matching existing
 * objects to new detections.
 */
public class OverlayTracker
{
    private final Logger logger = new Logger();
    private final Size previewSize;
    private final int[] colors;
    private final Vector<String> lastLabels;
    private final Vector<String> labels;
    private Bitmap bmp;
    private int[] pixels;

    public OverlayTracker(final Size previewSize, Vector<String> labels)
    {
        this.previewSize = previewSize;

        int[] colors = new int[2];
        int alpha = 100;
        colors[0] = Color.argb(alpha, 0, 0, 0);
        colors[1] = Color.argb(alpha, 128, 0, 0);
        this.colors = colors;
        this.labels = labels;
        this.lastLabels = new Vector<>();
    }

    public synchronized void trackResults(final Segmentor.Segmentation result, final long timestamp)
    {
        logger.i("Processing from %d", timestamp);
        processResults(timestamp, result);
    }

    public synchronized void draw(final Canvas canvas)
    {
        if (bmp != null)
        {
            final Matrix matrix = new Matrix();
            float multiplierX = canvas.getWidth() / (float) bmp.getWidth();
            float multiplierY = multiplierX * (float) previewSize.getWidth() / (float) previewSize.getHeight();
            matrix.postScale(multiplierX, multiplierY);
            matrix.postTranslate(0, 0);
            canvas.drawBitmap(bmp, matrix, new Paint(Paint.FILTER_BITMAP_FLAG));
        }
    }

    private void processResults(
            final long timestamp, final Segmentor.Segmentation result)
    {
        handleSegmentation(timestamp, result);
    }

    public Vector<String> getLastLabels()
    {
        return lastLabels;
    }

    private void handleSegmentation(final long timestamp, final Segmentor.Segmentation potential)
    {
        int width = potential.getWidth();
        int height = potential.getHeight();
        if (bmp == null)
        {
            bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        }
        if (pixels == null)
        {
            pixels = new int[bmp.getHeight() * bmp.getWidth()];
        }
        float[] resultPixels = potential.getPixels();

        int numClass = potential.getNumClass();
        int[] visitedLabels = new int[numClass];
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                int classNo = (int) resultPixels[j * height + i] == 255 ? 1 : 0;
                pixels[j * bmp.getWidth() + i] = colors[classNo];
                visitedLabels[classNo] = 1;
            }
        }

        lastLabels.clear();
        for (int i = 0; i < numClass; i++)
        {
            if (visitedLabels[i] == 1)
            {
                lastLabels.add(labels.get(i));
            }
        }

        bmp.setPixels(pixels, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
    }
}
