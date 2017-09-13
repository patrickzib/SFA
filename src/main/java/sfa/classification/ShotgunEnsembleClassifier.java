// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.EnsembleModel;
import sfa.transformation.ShotgunModel;

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
      EnsembleModel<ShotgunModel> bestModel = new EnsembleModel<>();
      int bestCorrectTesting = 0;
      int bestCorrectTraining = 0;

      for (boolean normMean : NORMALIZATION) {
        long startTime = System.currentTimeMillis();

        this.correctTraining = new AtomicInteger(0);

        EnsembleModel<ShotgunModel> models = fitEnsemble(exec, normMean, factor);

        // training score
        ShotgunModel model = models.getHighestScoringModel();
        if (DEBUG) {
          System.out.println("Shotgun Ensemble Training:\t" + model.length + "\tnormed: \t" + normMean);
          outputResult(this.correctTraining.get(), startTime, this.trainSamples.length);
        }

        int correctTesting = predictEnsemble(
            exec,
            models,
            this.testSamples,
            this.trainSamples // TODO move to shotgun model???
        );

        if (bestCorrectTraining < models.getHighestAccuracy()) {
          bestCorrectTesting = correctTesting;
          bestCorrectTraining = this.correctTraining.get();
          bestModel = models;
        }
        if (DEBUG) {
          System.out.println("");
        }
      }

      return new Score(
          "Shotgun Ensemble",
          1 - formatError(bestCorrectTesting, this.testSamples.length),
          1 - formatError(bestCorrectTraining, this.trainSamples.length),
          bestModel.model.get(0).normMean,
          bestModel.model.get(0).length);
    } finally {
      exec.shutdown();
    }
  }

  public int predictEnsemble(
      final ExecutorService executor,
      final EnsembleModel<ShotgunModel> model,
      final TimeSeries[] testSamples,
      final TimeSeries[] trainSamples) {
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
            final ShotgunModel m = model.model.get(i);
            final double accuracy = model.accuracy.get(i);

            usedLengths.add(m.length);

            Predictions p = predict(
                new ShotgunModel(m.length, m.normMean),
                testSamples,
                trainSamples);
            for (int a = 0; a < p.labels.length; a++) {
              testLabels[a].add(new Pair<>(p.labels[a], accuracy));
            }
          }
        }
      }
    });

    return score("Shotgun Ensemble", testSamples, startTime, testLabels, usedLengths);
  }
}