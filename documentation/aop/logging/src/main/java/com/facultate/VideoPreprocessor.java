package com.facultate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class VideoPreprocessor
{
    public byte[] openVideoFile(String path) throws IOException
    {
       return Files.readAllBytes(Paths.get(path));
    }

    public byte[] downsizeVideo(byte[] videoData)
    {
        for (int i = 0; i < videoData.length-3; i+=3)
            videoData[i] = (byte) ((videoData[i] + videoData[i+1] + videoData[i+2]) /3);

        return videoData;
    }

}
