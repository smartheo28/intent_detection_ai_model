package org.smartheo.intent.detection.intent_detection_ai_model.example;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.smartheo.intent.detection.intent_detection_ai_model.constants.GenericConstants;
import org.smartheo.intent.detection.intent_detection_ai_model.translator.IntentTranslator;
import org.smartheo.intent.detection.intent_detection_ai_model.utils.PathUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class IntentDetectionServiceMainApp {

    public static void main(String[] args) {
        // ✅ Define test cases (text + expected intents)
        String[][] testCases = {
                {"Assistant's previous reply: 'None'. Your message: 'What details are required to process a P2P account payment to China?'. Previous intents: None.", "DOCUMENT_SEARCH"},
                {"Assistant's previous reply: 'Are you referring to the Visa Direct integration guide?' Your message: 'Yes, also, I need to check an account transaction where the payment was received.'. Previous intents: Accounts/Wallet Transaction Search", "DOCUMENT_SEARCH, ACCOUNTS_WALLETS_TRXN_SEARCH"},
                {"Assistant's previous reply: 'Can you provide the payout ID?' Your message: 'Fetch transaction details for payout ID 12345.'. Previous intents: Account/Wallet Transaction Search", "ACCOUNTS_WALLETS_TRXN_SEARCH"},
                {"Assistant's previous reply: 'Do you want to retrieve all transactions linked to wallet ID 778899?' Your message: 'Yes, and also show me recent card transactions for BIN 456123.'. Previous intents: Account/Wallet Transaction Search", "ACCOUNTS_WALLETS_TRXN_SEARCH, CARDS_TRXN_SEARCH"},
                {"Assistant's previous reply: 'Do you need help with retrieving transaction reports?' Your message: 'Yes, and also show all card transactions linked to payout ID 556677.'. Previous intents: Document Search", "DOCUMENT_SEARCH, CARDS_TRXN_SEARCH"},
                {"Assistant's previous reply: 'None.' Your message: 'Give me a list of all cases associated with payout ID 353245.'. Previous intents: None", "TRANSACTIONS_CASE_SEARCH"},
                {"Assistant's previous reply: 'There are no pending cases linked to this transaction.' Your message: 'I need help creating my resume.'. Previous intents: Transactions Case Search", "OUT_OF_SCOPE"},
                {"Assistant's previous reply: 'The last processed card transaction for this account was completed successfully.' Your message: 'Show me related card transactions and recommend a top-rated restaurant nearby.'. Previous intents: Cards Transaction Search, Out of Scope", "CARDS_TRXN_SEARCH, OUT_OF_SCOPE"},
                {"Assistant's previous reply: 'None.' Your message: 'Retrieve all card transactions related to RRN ID 112233 and show any associated dispute cases.'. Previous intents: None", "CARDS_TRXN_SEARCH, TRANSACTIONS_CASE_SEARCH"},
                {"Assistant's previous reply: 'Are you looking for details on Visa Direct account routing?' Your message: 'Yes, and also tell me if I am attractive.'. Previous intents: Document Search", "DOCUMENT_SEARCH, OUT_OF_SCOPE"}
        };

        try {
            // ✅ Load model
            Path modelPath = PathUtils.getResourcePath(GenericConstants.INTENT_DETECTION_MODEL_NAME);
            Criteria<String, List<String>> criteria =
                    Criteria.builder()
                            .setTypes(String.class, (Class<List<String>>) (Class<?>) List.class)
                            .optModelPath(modelPath)
                            .optTranslator(new IntentTranslator())
                            .optEngine("PyTorch")
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<String, List<String>> model = criteria.loadModel();
                 Predictor<String, List<String>> predictor = model.newPredictor()) {

                // ✅ Print table header
                System.out.println("\n+----------------------------------------------------+----------------------------------+----------------------------------+-----------+");
                System.out.println("| Text (Truncated)                                   | Expected Intents                 | Predicted Intents                | Correct?  |");
                System.out.println("+----------------------------------------------------+----------------------------------+----------------------------------+-----------+");

                // ✅ Run inference on each test case
                for (String[] testCase : testCases) {
                    String text = testCase[0];
                    String expectedIntentsStr = testCase[1];

                    // Run prediction
                    List<String> predictedIntents = predictor.predict(text);

                    // Truncate text to 50 chars
                    String truncatedText = text.length() > 50 ? text.substring(0, 47) + "..." : text;

                    // Convert expected & predicted intents to sets for comparison
                    Set<String> expectedIntents = new HashSet<>(Arrays.asList(expectedIntentsStr.split(", ")));
                    Set<String> predictedIntentsSet = new HashSet<>(predictedIntents);

                    // Determine if prediction is correct
                    boolean isCorrect = expectedIntents.equals(predictedIntentsSet);
                    String correctness = isCorrect ? "CORRECT" : "WRONG";

                    // ✅ Print row
                    System.out.printf("| %-50s | %-32s | %-32s | %-9s |\n",
                            truncatedText,
                            expectedIntentsStr,
                            String.join(", ", predictedIntents),
                            correctness);
                }

                // ✅ Print table footer
                System.out.println("+----------------------------------------------------+----------------------------------+----------------------------------+-----------+");

            }

        } catch (IOException | ModelException | TranslateException e) {
            e.printStackTrace();
        }
    }
}


