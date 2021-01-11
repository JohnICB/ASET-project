package info.uaic.ro.segmentation;

import android.graphics.Bitmap;

import java.util.Vector;

/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Segmentor
{
    Segmentation segmentImage(Bitmap bitmap);

    Vector<String> getLabels();

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    class Segmentation
    {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final float[] pixels;
        private final int numClass;
        private final int width;
        private final int height;
        private final long inferenceTime;
        private final long nativeTime;

        public Segmentation(final float[] pixels2, final int numClass, final int width, final int height,
                            final long inferenceTime, final long nativeTime)
        {
            this.pixels = pixels2;
            this.numClass = numClass;
            this.width = width;
            this.height = height;
            this.inferenceTime = inferenceTime;
            this.nativeTime = nativeTime;
        }

        public float[] getPixels()
        {
            return pixels;
        }

        public int getWidth()
        {
            return width;
        }

        public int getHeight()
        {
            return height;
        }

        public long getInferenceTime()
        {
            return inferenceTime;
        }

        public long getNativeTime()
        {
            return nativeTime;
        }

        public int getNumClass()
        {
            return numClass;
        }
    }
}
