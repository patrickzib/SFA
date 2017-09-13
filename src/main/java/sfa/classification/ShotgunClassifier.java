// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.EnsembleModel;
import sfa.transformation.ShotgunModel;

/**
 * The Shotgun Classifier as published in:
 * <p>
 * Schäfer, P.: Towards time series classification without human preprocessing.
 * In Machine Learning and Data Mining in Pattern Recognition,
 * pages 228–242. Springer, 2014.
 */
public class ShotgunClassifier extends Classifier {

  public ShotgunClassifier(TimeSeries[] train, TimeSeries[] test) {
    super(train, test);
  }

  public Score eval() {
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    try {

      // Shotgun Distance
      ShotgunModel bestModel = new ShotgunModel(0, false, trainSamples);
      int bestCorrectTesting = 0;
      int bestCorrectTraining = 0;

      for (boolean normMean : NORMALIZATION) {
        long startTime = System.currentTimeMillis();

        this.correctTraining = new AtomicInteger(0);

        EnsembleModel<ShotgunModel> models = fitEnsemble(exec, this.trainSamples, normMean, 1.0);

        // training score
        ShotgunModel model = models.getHighestScoringModel();
        if (DEBUG) {
          System.out.println("Shotgun Training:\t" + model.length + "\tnormed: \t" + normMean);
          outputResult(this.correctTraining.get(), startTime, model.samples.length);
        }

        // Classify: testing score
        int correctTesting = predict(
                model,
                this.testSamples).correct.get();

        if (bestCorrectTraining < models.getHighestAccuracy()) {
          bestCorrectTesting = correctTesting;
          bestCorrectTraining = this.correctTraining.get();
          bestModel = model;
        }
        if (DEBUG) {
          System.out.println("");
        }
      }
      return new Score(
          "Shotgun",
          1 - formatError(bestCorrectTesting, this.testSamples.length),
          1 - formatError(bestCorrectTraining, this.trainSamples.length),
          bestModel.normMean,
          bestModel.length);
    } finally {
      exec.shutdown();
    }
  }


  public EnsembleModel fitEnsemble(
      final ExecutorService exec,
      final TimeSeries[] samples,
      final boolean normMean,
      final double factor) {
    int minWindowLength = 5;
    int maxWindowLength = MAX_WINDOW_LENGTH;
    for (TimeSeries ts : samples) {
      maxWindowLength = Math.min(ts.getLength(), maxWindowLength);
    }

    ArrayList<Integer> windows = new ArrayList<>();
    for (int windowLength = maxWindowLength; windowLength >= minWindowLength; windowLength--) {
      windows.add(windowLength);
    }

    return fit(windows.toArray(new Integer[]{}), normMean, samples, factor, exec);
  }

  public EnsembleModel fit(
      final Integer[] allWindows,
      final boolean normMean,
      final TimeSeries[] samples,
      final double factor,
      ExecutorService exec) {
    final List<Score> results = new ArrayList<>(allWindows.length);
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      Score bestScore = new Score("Shotgun", 0, 0, normMean, 0);
      final Object sync = new Object();

      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < allWindows.length; i++) {
          if (i % threads == id) {
            Predictions p = predict(
                new ShotgunModel(allWindows[i], normMean, samples),
                samples
            );

            Score score = new Score("Shotgun", p.correct.get(), p.correct.get(), normMean, allWindows[i]);

            // keep best scores
            synchronized (sync) {
              if (this.bestScore.compareTo(score) <= 0) {
                ShotgunClassifier.this.correctTraining.set((int) score.training);
                this.bestScore = score;
              }
            }

            // add to ensemble
            if (score.training >= ShotgunClassifier.this.correctTraining.get() * factor) { // all with same score
              synchronized (results) {
                results.add(score);
              }
            }
          }
        }
      }
    });

    // sort descending
    Collections.sort(results, Collections.reverseOrder());

    // only keep best scores
    List<ShotgunModel> model = new ArrayList<>();
    List<Double> accuracy = new ArrayList<>();

    for (int i = 0; i < results.size(); i++) {
      final Score score = results.get(i);
      if (score.training >= this.correctTraining.get() * factor) { // all with same score
        model.add(new ShotgunModel(score.windowLength, score.normed, samples));
        accuracy.add(score.training);
      }
    }

    return new EnsembleModel<ShotgunModel>(model, accuracy);
  }

  public Predictions predict(
      final ShotgunModel model,
      final TimeSeries[] testSamples) {

    final Predictions p = new Predictions(new String[testSamples.length], 0);

    // calculate means and stds for each sample
    final double[][] means = new double[model.samples.length][];
    final double[][] stds = new double[model.samples.length][];
    calcMeansStds(model.length, model.samples, means, stds, model.normMean);

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < testSamples.length; i++) {
          if (i % BLOCKS == id) {
            final TimeSeries query = testSamples[i];

            double distanceTo1NN = Double.MAX_VALUE;
            String predictedLabel = "";

            int wQueryLen = Math.min(query.getLength(), model.length);
            TimeSeries[] disjointWindows = query.getDisjointSequences(wQueryLen, model.normMean); // possible without copying!?

            // perform a 1-NN search
            for (int j = 0; j < model.samples.length; j++) {
              TimeSeries ts = model.samples[j];

              // Shotgun Distance
              if (ts != query) {
                double totalDistance = 0.0;

                for (final TimeSeries q : disjointWindows) {
                  double resultDistance = distanceTo1NN;

                  // calculate euclidean distances for each sliding window
                  for (int ww = 0, end = ts.getLength() - model.length + 1; ww < end; ww++) { // faster than reevaluation in for loop
                    double distance = getEuclideanDistance(ts, q, means[j][ww], stds[j][ww], resultDistance, ww);
                    resultDistance = Math.min(distance, resultDistance);
                  }
                  totalDistance += resultDistance;

                  // pruning on distance
                  if (totalDistance > distanceTo1NN) {
                    break;
                  }
                }

                // choose minimum
                if (totalDistance < distanceTo1NN) {
                  predictedLabel = ts.getLabel();
                  distanceTo1NN = totalDistance;
                }
              }
            }

            // check if the prediction is correct
            p.labels[i] = predictedLabel;
            if (testSamples[i].getLabel().equals(p.labels[i])) {
              p.correct.incrementAndGet();
            }
          }
        }
      }
    });

    return p;
  }

  public static double getEuclideanDistance(
      TimeSeries ts,
      TimeSeries q,
      double meanTs,
      double stdTs,
      double minValue,
      int w
  ) {

    double distance = 0.0;
    double[] tsData = ts.getData();
    double[] qData = q.getData();

    for (int ww = 0; ww < qData.length; ww++) {
      double value1 = (tsData[w + ww] - meanTs) * stdTs;
      double value = qData[ww] - value1;
      distance += value * value;

      // early abandoning
      if (distance >= minValue) {
        return Double.MAX_VALUE;
      }
    }

    return distance;
  }

  public static void calcMeansStds(
      final int windowLength,
      final TimeSeries[] trainSamples,
      final double[][] means,
      final double[][] stds,
      boolean normMean) {
    for (int i = 0; i < trainSamples.length; i++) {
      int w = Math.min(windowLength, trainSamples[i].getLength());
      means[i] = new double[trainSamples[i].getLength() - w + 1];
      stds[i] = new double[trainSamples[i].getLength() - w + 1];
      TimeSeries.calcIncrementalMeanStddev(w, trainSamples[i].getData(), means[i], stds[i]);
      for (int j = 0; j < stds[i].length; j++) {
        stds[i][j] = (stds[i][j] > 0 ? 1.0 / stds[i][j] : 1.0);
        means[i][j] = normMean ? means[i][j] : 0;
      }

    }
  }
}