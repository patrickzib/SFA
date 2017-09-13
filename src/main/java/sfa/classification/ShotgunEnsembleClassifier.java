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

  public static double factor = 0.92;

  public ShotgunEnsembleClassifier(TimeSeries[] train, TimeSeries[] test) {
    super(train, test);
  }

  @Override
  public Score eval() {
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    try {
      Score totalBestScore = new Score("not initialized", -1, -1, false, -1);
      int bestCorrectTesting = 0;
      int bestCorrectTraining = 0;

      for (boolean normMean : NORMALIZATION) {
        long startTime = System.currentTimeMillis();

        this.correctTraining = new AtomicInteger(0);

        List<Score> scores = fitEnsemble(exec, this.trainSamples, normMean, factor);

        // training score
        Score bestScore = scores.get(0);
        if (DEBUG) {
          System.out.println("Shotgun Ensemble Training:\t" + bestScore.windowLength + "\tnormed: \t" + normMean);
          outputResult(this.correctTraining.get(), startTime, this.trainSamples.length);
        }

        // Classify: testing score
        int correctTesting = predictEnsemble(exec, scores, this.testSamples, this.trainSamples);

        if (bestCorrectTraining < bestScore.training) {
          bestCorrectTesting = correctTesting;
          bestCorrectTraining = this.correctTraining.get();
          totalBestScore = bestScore;
        }
        if (DEBUG) {
          System.out.println("");
        }
      }

      return new Score(
          "Shotgun Ensemble",
          1 - formatError(bestCorrectTesting, this.testSamples.length),
          1 - formatError(bestCorrectTraining, this.trainSamples.length),
          totalBestScore.normed,
          totalBestScore.windowLength);
    } finally {
      exec.shutdown();
    }
  }

  public int predictEnsemble(
      final ExecutorService executor,
      final List<Score> results,
      final TimeSeries[] testSamples,
      final TimeSeries[] trainSamples) {
    long startTime = System.currentTimeMillis();

    @SuppressWarnings("unchecked")
    final List<Pair<String, Double>>[] testLabels = new List[testSamples.length];
    for (int i = 0; i < testLabels.length; i++) {
      testLabels[i] = new ArrayList<>();
    }

    final List<Integer> usedLengths = new ArrayList<>(results.size());

    // parallel execution
    ParallelFor.withIndex(executor, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < results.size(); i++) {
          if (i % threads == id) {
            final Score score = results.get(i);
            if (score.training >= ShotgunEnsembleClassifier.this.correctTraining.get() * factor) { // all with same score
              usedLengths.add(score.windowLength);

              Predictions p = predict(score.windowLength, score.normed, testSamples, trainSamples);
              for (int a = 0; a < p.labels.length; a++) {
                testLabels[a].add(new Pair<>(p.labels[a], score.training));
              }
            }
          }
        }
      }
    });

    return score("Shotgun Ensemble", testSamples, startTime, testLabels, usedLengths);
  }
}