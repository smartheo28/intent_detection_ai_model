package org.smartheo.intent.detection.intent_detection_ai_model.utils;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

public class PathUtils {

    public static Path getResourcePath(String resource) throws IOException {
        URL resourceUrl = Thread.currentThread().getContextClassLoader().getResource(resource);
        if (resourceUrl == null) {
            throw new IOException("Resource not found: " + resource);
        }
        return Paths.get(resourceUrl.getPath());
    }
}
