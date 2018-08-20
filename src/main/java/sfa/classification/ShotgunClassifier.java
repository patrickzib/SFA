// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;

/**
 * The Shotgun Classifier as published in:
 * <p>
 * Schäfer, P.: Towards time series classification without human preprocessing.
 * In Machine Learning and Data Mining in Pattern Recognition,
 * pages 228–242. Springer, 2014.
 */
public class ShotgunClassifier extends Classifier {

  // the trained boss
  ShotgunModel model;

  public static int MAX_WINDOW_LENGTH = 250;

  public ShotgunClassifier() {
    super();
  }

  public static class ShotgunModel extends Model {

    public ShotgunModel(){}

    public ShotgunModel(
            boolean normed,
            int windowLength,
            TimeSeries[] samples
    ) {
      super("Shotgun", -1, 1, -1, 1, normed, windowLength);
      this.samples = samples;
    }

    public TimeSeries[] samples; // the train samples needed for 1-NN classification
  }

  public Score eval(
          final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {

    long startTime = System.currentTimeMillis();

    Score score = fit(trainSamples);

    // training score
    if (DEBUG) {
      System.out.println(score.toString());
      outputResult(score.training, startTime, trainSamples.length);
    }

    // Classify: testing score
    int correctTesting = score(testSamples).correct.get();

    if (DEBUG) {
      System.out.println("Shotgun Testing:\t");
      outputResult(correctTesting, startTime, testSamples.length);
      System.out.println("");
    }

    return new Score(
            "Shotgun",
            correctTesting, testSamples.length,
            score.training, trainSamples.length,
            score.windowLength);
  }


  @Override
  public Score fit(final TimeSeries[] trainSamples) {
    Score bestScore = null;
    int bestCorrectTraining = 0;

    for (boolean normMean : NORMALIZATION) {
      //long startTime = System.currentTimeMillis();

      // train the shotgun models for different window lengths
      ShotgunModel model = fitEnsemble(trainSamples, normMean, 1.0).getHighestScoringModel();
      Score score = model.score;

      if (this.model == null || bestCorrectTraining < score.training) {
        bestCorrectTraining = score.training;
        bestScore = score;
        this.model = model;
      }
    }

    // return score
    return bestScore;
  }

  @Override
  public Predictions score(final TimeSeries[] testSamples) {
    Double[] labels = predict(testSamples);
    return evalLabels(testSamples, labels);
  }

  protected Ensemble<ShotgunModel> fitEnsemble(
          final TimeSeries[] trainSamples,
          final boolean normMean,
          final double factor) {

    int minWindowLength = 5;
    int maxWindowLength = getMax(trainSamples, MAX_WINDOW_LENGTH);
    Integer[] windows = getWindowsBetween(minWindowLength, maxWindowLength);

    final AtomicInteger correctTraining = new AtomicInteger(0);

    final List<ShotgunModel> results = new ArrayList<>(windows.length);
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < windows.length; i++) {
          if (i % threads == id) {
            ShotgunModel model = new ShotgunModel(normMean, windows[i], trainSamples);
            Double[] labels = predict(model, trainSamples);
            Predictions p = evalLabels(trainSamples, labels);

            model.score = new Score(model.name, -1, 1, p.correct.get(), trainSamples.length, windows[i]);

            // keep best scores
            synchronized (correctTraining) {
              if (model.score.training > correctTraining.get()) {
                correctTraining.set(model.score.training);
              }

              // add to ensemble if train-score is within factor to the best score
              if (model.score.training >= correctTraining.get() * factor) {
                results.add(model);
              }
            }
          }
        }
      }
    });

    // returns the ensemble based on the best window-lengths within factor
    return filterByFactor(results, correctTraining.get(), factor);
  }

  @Override
  public Double[] predict(final TimeSeries[] testSamples) {
    return predict(this.model, testSamples);
  }

  protected Double[] predict(ShotgunModel model, final TimeSeries[] testSamples) {

    final Double[] p = new Double[testSamples.length];

    // calculate means and stds for each sample
    final double[][] means = new double[model.samples.length][];
    final double[][] stds = new double[model.samples.length][];
    calcMeansStds(model.windowLength, model.samples, means, stds, model.normed);

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < testSamples.length; i++) {
          if (i % BLOCKS == id) {
            final TimeSeries query = testSamples[i];

            double distanceTo1NN = Double.MAX_VALUE;

            int wQueryLen = Math.min(query.getLength(), model.windowLength);
            TimeSeries[] disjointWindows = query.getDisjointSequences(wQueryLen, model.normed); // possible without copying!?

            // perform a 1-NN search
            for (int j = 0; j < model.samples.length; j++) {
              TimeSeries ts = model.samples[j];

              // Shotgun Distance
              if (ts != query) {
                double totalDistance = 0.0;

                for (final TimeSeries q : disjointWindows) {
                  double resultDistance = distanceTo1NN;

                  // calculate euclidean distances for each sliding window
                  for (int ww = 0, end = ts.getLength() - model.windowLength + 1; ww < end; ww++) { // faster than reevaluation in for loop
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
                  p[i] = ts.getLabel();
                  distanceTo1NN = totalDistance;
                }
              }
            }
          }
        }
      }
    });

    return p;
  }

  protected static double getEuclideanDistance(
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

  protected static void calcMeansStds(
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