package org.smartheo.intent.detection.intent_detection_ai_model.translator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.smartheo.intent.detection.intent_detection_ai_model.constants.GenericConstants;
import org.smartheo.intent.detection.intent_detection_ai_model.utils.PathUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IntentTranslator implements Translator<String, List<String>> {

    private HuggingFaceTokenizer tokenizer;
    private static final int MAX_LENGTH = 512;
    private static final float THRESHOLD = 0.05f; // Same threshold as Python

    // List of intent labels (Must match training labels order)
    private final List<String> intentLabels = Arrays.asList(
            "DOCUMENT_SEARCH",
            "ACCOUNTS_WALLETS_TRXN_SEARCH",
            "CARDS_TRXN_SEARCH",
            "TRANSACTIONS_CASE_SEARCH",
            "OUT_OF_SCOPE"
    );

    public IntentTranslator() {
        try {
            tokenizer = HuggingFaceTokenizer.newInstance(PathUtils.getResourcePath(GenericConstants.INTENT_DETECTION_MODEL_TOKENIZER));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load tokenizer", e);
        }
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        Encoding encoding = tokenizer.encode(input);

        long[] inputIds = padOrTruncate(encoding.getIds(), MAX_LENGTH);
        long[] attentionMask = padOrTruncate(encoding.getAttentionMask(), MAX_LENGTH);

        NDArray inputIdsArray = ctx.getNDManager().create(inputIds).reshape(1, MAX_LENGTH);
        NDArray attentionMaskArray = ctx.getNDManager().create(attentionMask).reshape(1, MAX_LENGTH);

        return new NDList(inputIdsArray, attentionMaskArray);
    }

    @Override
    public List<String> processOutput(TranslatorContext ctx, NDList list) {
        NDArray logits = list.get(0); // Raw model output
        // Apply Sigmoid Activation Manually
        NDArray probabilities = logits.mul(-1).exp().add(1).pow(-1);// ✅ Correct sigmoid function

        // Apply thresholding (0.05) to get binary predictions
        NDArray predictions = probabilities.gt(THRESHOLD); // Greater than threshold

        // Convert NDArray to boolean array
        boolean[] predictedArray = predictions.toBooleanArray();

        // Map indices to intent labels
        List<String> predictedIntents = new ArrayList<>();
        for (int i = 0; i < predictedArray.length; i++) {
            if (predictedArray[i]) {
                predictedIntents.add(intentLabels.get(i));
            }
        }

        return predictedIntents;
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    private long[] padOrTruncate(long[] tokens, int targetLength) {
        if (tokens.length == targetLength) {
            return tokens;
        } else if (tokens.length > targetLength) {
            return Arrays.copyOf(tokens, targetLength);
        } else {
            long[] padded = new long[targetLength];
            System.arraycopy(tokens, 0, padded, 0, tokens.length);
            return padded;
        }
    }
}