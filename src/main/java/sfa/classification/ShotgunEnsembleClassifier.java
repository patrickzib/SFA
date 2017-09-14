// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;

/**
 * The Shotgun Ensemble Classifier as published in:
 * <p>
 * Schäfer, P.: Towards time series classification without human preprocessing.
 * In Machine Learning and Data Mining in Pattern Recognition,
 * pages 228–242. Springer, 2014.
 */
public class ShotgunEnsembleClassifier extends ShotgunClassifier {

  // default training parameters
  public static double factor = 0.92;

  // the trained boss
  Ensemble<ShotgunModel> model;

  public ShotgunEnsembleClassifier(TimeSeries[] train, TimeSeries[] test) {
    super(train, test);
  }

  @Override
  public Score eval() {

    long startTime = System.currentTimeMillis();

    Score score = fit(this.trainSamples);

    if (DEBUG) {
      System.out.println(score.toString());
      outputResult((int) score.training, startTime, this.testSamples.length);
      System.out.println("");
    }

    // Classify: testing score
    int correctTesting = predict(this.testSamples).correct.get();

    return new Score(
        "Shotgun Ensemble",
        1 - formatError(correctTesting, this.testSamples.length),
        1 - formatError((int) score.training, this.trainSamples.length),
        score.windowLength);
  }

  public Score fit(final TimeSeries[] samples) {

    Score bestScore = null;
    int bestCorrectTraining = 0;

    for (boolean normMean : NORMALIZATION) {
      // train the shotgun models for different window lengths
      Ensemble<ShotgunModel> model = fitEnsemble(samples, normMean, factor);
      Score score = model.getHighestScoringModel().score;
      Predictions pred = predictEnsemble(model, samples);

      if (model == null || bestCorrectTraining <= pred.correct.get()) {
        bestCorrectTraining = pred.correct.get();
        bestScore = score;
        this.model = model;
      }
    }

    // return score
    return bestScore;
  }

  public Predictions predict(final TimeSeries[] testSamples) {
    return predictEnsemble(this.model, testSamples);
  }

  protected Predictions predictEnsemble(
      final Ensemble<ShotgunModel> model,
      final TimeSeries[] testSamples) {
    long startTime = System.currentTimeMillis();

    @SuppressWarnings("unchecked")
    final List<Pair<String, Double>>[] testLabels = new List[testSamples.length];
    for (int i = 0; i < testLabels.length; i++) {
      testLabels[i] = new ArrayList<>();
    }

    final List<Integer> usedLengths = new ArrayList<>(model.model.size());

    // parallel execution
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < model.model.size(); i++) {
          if (i % threads == id) {
            final ShotgunModel score = model.model.get(i);

            usedLengths.add(score.windowLength);

            Predictions p = predict(score, testSamples);

            for (int a = 0; a < p.labels.length; a++) {
              testLabels[a].add(new Pair<>(p.labels[a], score.score.training));
            }
          }
        }
      }
    });

    return score("Shotgun Ensemble", testSamples, startTime, testLabels, usedLengths);
  }
}