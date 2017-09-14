// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.Collections;
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

  public ShotgunClassifier(TimeSeries[] train, TimeSeries[] test) {
    super(train, test);
  }

  public class ShotgunModel extends Model {
    public ShotgunModel(
        boolean normed,
        int windowLength,
        TimeSeries[] samples
    ) {
      super("Shotgun", -1, -1, normed, windowLength);
      this.samples = samples;
    }

    public TimeSeries[] samples;
  }

  public Score eval() {

    long startTime = System.currentTimeMillis();

    Score score = fit(this.trainSamples);

    // training score
    if (DEBUG) {
      System.out.println(score.toString());
      outputResult((int) score.training, startTime, trainSamples.length);
    }

    // Classify: testing score
    int correctTesting = predict(this.testSamples).correct.get();

    if (DEBUG) {
      System.out.println("Shotgun Testing:\t");
      outputResult(correctTesting, startTime, this.testSamples.length);
      System.out.println("");
    }

    return new Score(
        "Shotgun",
        1 - formatError(correctTesting, this.testSamples.length),
        1 - formatError((int) score.training, this.trainSamples.length),
        score.windowLength);
  }


  public Score fit(final TimeSeries[] samples) {
    Score bestScore = null;
    int bestCorrectTraining = 0;

    for (boolean normMean : NORMALIZATION) {
      long startTime = System.currentTimeMillis();

      // train the shotgun models for different window lengths
      ShotgunModel model = fitEnsemble(samples, normMean, 1.0).getHighestScoringModel();
      Score score = model.score;

      if (bestCorrectTraining < score.training) {
        bestCorrectTraining = (int) score.training;
        bestScore = score;
        this.model = model;
      }
    }

    // return score
    return bestScore;
  }

  protected Ensemble<ShotgunModel> fitEnsemble(
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
    Integer[] allWindows = windows.toArray(new Integer[]{});

    final AtomicInteger correctTraining = new AtomicInteger(0);

    final List<ShotgunModel> results = new ArrayList<>(allWindows.length);
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      ShotgunModel bestModel = null;
      final Object sync = new Object();

      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < allWindows.length; i++) {
          if (i % threads == id) {
            ShotgunModel model = new ShotgunModel(normMean, allWindows[i], samples);
            Predictions p = predict(
                model,
                samples
            );

            model.score = new Score(model.name, -1, p.correct.get(), allWindows[i]);

            // keep best scores
            synchronized (sync) {
              if (this.bestModel == null || this.bestModel.compareTo(model) <= 0) {
                correctTraining.set((int) model.score.training);
                this.bestModel = model;
              }
            }

            // add to ensemble
            if (model.score.training >= correctTraining.get() * factor) { // all with higher score
              synchronized (results) {
                results.add(model);
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

    for (int i = 0; i < results.size(); i++) {
      final ShotgunModel m = results.get(i);
      if (m.score.training >= correctTraining.get() * factor) { // all with same score
        model.add(m);
      }
    }

    return new Ensemble<>(results);
  }


  public Predictions predict(final TimeSeries[] testSamples) {
    return predict(this.model, testSamples);
  }


  protected Predictions predict(
      final ShotgunModel model,
      final TimeSeries[] testSamples) {

    final Predictions p = new Predictions(new String[testSamples.length], 0);

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
            String predictedLabel = "";

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