// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    try {
      Ensemble<ShotgunModel> bestScore = new Ensemble<>();
      int bestCorrectTesting = 0;
      int bestCorrectTraining = 0;

      for (boolean normMean : NORMALIZATION) {
        long startTime = System.currentTimeMillis();

        Score score = fit(exec, this.trainSamples, normMean);

        // training score
        if (DEBUG) {
          System.out.println(score.toString());
          outputResult((int)score.training, startTime, this.trainSamples.length);
        }

        int correctTesting = predict(exec, this.testSamples).correct.get();

        if (bestCorrectTraining < score.training) {
          bestCorrectTesting = correctTesting;
          bestCorrectTraining = (int)score.training;
          bestScore = model;
        }
        if (DEBUG) {
          System.out.println("");
        }
      }

      return new Score(
          "Shotgun Ensemble",
          1 - formatError(bestCorrectTesting, this.testSamples.length),
          1 - formatError(bestCorrectTraining, this.trainSamples.length),
          bestScore.model.get(0).windowLength);
    } finally {
      exec.shutdown();
    }
  }

  public Score fit(
      final ExecutorService exec,
      final TimeSeries[] samples,
      final boolean normMean) {

    // train the shotgun models for different window lengths
    this.model = fitEnsemble(samples, normMean, 1.0, exec);

    // return score
    return model.getHighestScoringModel().score;
  }

  public Predictions predict(
      final ExecutorService executor,
      final TimeSeries[] testSamples) {
    return predictEnsemble(executor, this.model, testSamples);
  }

  protected Predictions predictEnsemble(
      final ExecutorService executor,
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
    ParallelFor.withIndex(executor, threads, new ParallelFor.Each() {
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