package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.inference.Predictor;
import ai.djl.translate.Batchifier;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * ===========================
 * Custom Input Class for Track Prediction
 * ===========================
 */
class TrackInput {

    /**
     * Feature array: length = 11 (5 avgWire + 5 slope + superlayer of missing cluster)
     */
    float[] features;

    /**
     * Constructor: validates length and normalizes input features.
     *
     * @param features float array of length 11
     */
    public TrackInput(float[] features) {
        if (features.length != 11) {
            throw new IllegalArgumentException("Expected 11 features");
        }
        this.features = normalize(features);
    }

    /**
     * Normalize features.
     * - First 5 values (avgWire) are divided by 112.0
     * - Next 5 values (slope) are unchanged
     * - Last value is superlayer of missing cluster divided by 6.0
     *
     * @param feats input feature array
     * @return normalized feature array
     */
    private float[] normalize(float[] feats) {
        float[] norm = new float[11];

        // Normalize avgWire
        for (int i = 0; i < 5; i++) {
            norm[i] = feats[i] / 112.0f;
        }

        // Copy slope features as-is
        for (int i = 5; i < 10; i++) {
            norm[i] = feats[i];
        }

        norm[10] =  feats[10]/6.0f;

        return norm;
    }
}

/**
 * ===========================
 * Main Inference Program
 * ===========================
 */
public class Main {

    public static void main(String[] args) {

        // -----------------------------
        // 1. Translator: TrackInput -> Float (track probability)
        // -----------------------------
        Translator<TrackInput, Float> translator = new Translator<TrackInput, Float>() {

            @Override
            public NDList processInput(TranslatorContext ctx, TrackInput input) {
                NDManager manager = ctx.getNDManager();
                // Shape: (1, 12) for single sample
                NDArray x = manager.create(input.features).reshape(1, input.features.length);
                return new NDList(x);
            }

            @Override
            public Float processOutput(TranslatorContext ctx, NDList list) {
                NDArray result = list.get(0); // Shape: (1,)
                return result.toFloatArray()[0]; // Extract single predicted value
            }

            @Override
            public Batchifier getBatchifier() {
                return null; // Single-sample inference (no batching)
            }
        };

        // -----------------------------
        // 2. Define model loading criteria
        // -----------------------------
        Criteria<TrackInput, Float> criteria = Criteria.builder()
                .setTypes(TrackInput.class, Float.class)
                .optModelPath(Paths.get("nets/mlp_default.pt"))  // TorchScript model path
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        // -----------------------------
        // 3. Load model and run inference
        // -----------------------------
        try (ZooModel<TrackInput, Float> model = criteria.loadModel();
             Predictor<TrackInput, Float> predictor = model.newPredictor()) {

            // Example input (11 float features)
            float[] exampleFeatures = new float[]{
                    21.7500f,19.0000f,18.4286f,15.3333f,14.5000f,-0.2316f,-0.2478f,-0.2987f,-0.3164f,-0.2833f,1.0f
            };

            TrackInput input = new TrackInput(exampleFeatures);

            Float probability = predictor.predict(input);
            System.out.printf("Predicted track probability: %.4f%n", probability);

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException("Model inference failed", e);
        }
    }
}