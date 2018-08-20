// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.BOSS;
import sfa.transformation.BOSS.BagOfPattern;

import com.carrotsearch.hppc.cursors.IntIntCursor;

/**
 * The Bag-of-SFA-Symbols Ensemble Classifier as published in
 * <p>
 * Schäfer, P.: The boss is concerned with time series classification
 * in the presence of noise. DMKD (2015)
 */
public class BOSSEnsembleClassifier extends Classifier {
  // default training parameters
  public static double factor = 0.92;

  public static int maxF = 16;
  public static int minF = 6;
  public static int maxS = 4;

  public static int MAX_WINDOW_LENGTH = 250;

  // the trained weasel
  public Ensemble<BOSSModel> model;

  public BOSSEnsembleClassifier() {
    super();
  }

  public static class BOSSModel extends Model {

    public BOSSModel(){}

    public BOSSModel(
        boolean normed,
        int windowLength) {
      super("BOSS", -1, 1, -1, 1, normed, windowLength);
    }

    // the BOSS representation for each train sample
    public BagOfPattern[] bag;

    // the trained BOSS transformation
    public BOSS boss;

    // the best number of Fourier values to be used
    public int features;
  }

  @Override
  public Score eval(
      final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
    long startTime = System.currentTimeMillis();

    Score score = fit(trainSamples);

    if (DEBUG) {
      System.out.println(score.toString());
      outputResult(score.training, startTime, testSamples.length);
      System.out.println("");
    }

    // Classify: testing score
    int correctTesting = score(testSamples).correct.get();

    return new Score(
        "BOSS Ensemble",
        correctTesting, testSamples.length,
        score.training, trainSamples.length,
        score.windowLength);
  }

  @Override
  public Score fit(final TimeSeries[] trainSamples) {
    // generate test train/split for cross-validation
    generateIndices(trainSamples);

    Score bestScore = null;
    int bestCorrectTraining = 0;

    int minWindowLength = 10;
    int maxWindowLength = getMax(trainSamples, MAX_WINDOW_LENGTH);
    Integer[] windows = getWindowsBetween(minWindowLength, maxWindowLength);

    for (boolean normMean : NORMALIZATION) {
      // train the shotgun models for different window lengths
      Ensemble<BOSSModel> model = fitEnsemble(windows, normMean, trainSamples);
      Double[] labels = predict(model, trainSamples);
      Predictions pred = evalLabels(trainSamples, labels);

      if (model == null || bestCorrectTraining < pred.correct.get()) {
        bestCorrectTraining = pred.correct.get();
        bestScore = model.getHighestScoringModel().score;
        bestScore.training = pred.correct.get();
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


  @Override
  public Double[] predict(final TimeSeries[] testSamples) {
    return predict(this.model, testSamples);
  }


  protected Ensemble<BOSSModel> fitEnsemble(Integer[] windows,
                                            boolean normMean,
                                            TimeSeries[] samples) {

    final ArrayList<BOSSModel> results = new ArrayList<>(windows.length);
    final AtomicInteger correctTraining = new AtomicInteger(0);

    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < windows.length; i++) {
          if (i % threads == id) {
            BOSSModel model = new BOSSModel(normMean, windows[i]);
            try {
              BOSS boss = new BOSS(maxF, maxS, windows[i], model.normed);
              int[][] words = boss.createWords(samples);

              for (int f = minF; f <= maxF; f += 2) {

                BagOfPattern[] bag = boss.createBagOfPattern(words, samples, f);

                Predictions p = predict(bag, bag);

                if (p.correct.get() > model.score.training) {
                  model.score.training = p.correct.get();
                  model.features = f;
                  model.boss = boss;
                  model.bag = bag;

                  if (p.correct.get() == samples.length) {
                    break;
                  }
                }
              }
            } catch (Exception e) {
              e.printStackTrace();
            }

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


  protected Predictions predict(
      final BagOfPattern[] bagOfPatternsTestSamples,
      final BagOfPattern[] bagOfPatternsTrainSamples) {

    Predictions p = new Predictions(new Double[bagOfPatternsTestSamples.length], 0);

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
          if (i % BLOCKS == id) {
            long minDistance = Integer.MAX_VALUE;

            // Distance if there is no matching word
            double noMatchDistance = 0.0;
            for (IntIntCursor key : bagOfPatternsTestSamples[i].bag) {
              noMatchDistance += key.value * key.value;
            }

            nnSearch:
            for (int j = 0; j < bagOfPatternsTrainSamples.length; j++) {
              if (bagOfPatternsTestSamples[i] != bagOfPatternsTrainSamples[j]) {
                // determine distance
                long distance = 0;
                for (IntIntCursor key : bagOfPatternsTestSamples[i].bag) {
                  long buf = key.value - bagOfPatternsTrainSamples[j].bag.get(key.key);
                  distance += buf * buf;

                  if (distance >= minDistance) {
                    continue nnSearch;
                  }
                }

                // update nearest neighbor
                if (distance != noMatchDistance && distance < minDistance) {
                  minDistance = distance;
                  p.labels[i] = bagOfPatternsTrainSamples[j].label;
                }
              }
            }

            // check if the prediction is correct
            if (compareLabels(bagOfPatternsTestSamples[i].label, p.labels[i])) {
              p.correct.incrementAndGet();
            }
          }
        }
      }
    });

    return p;
  }

  protected Double[] predict(final Ensemble<BOSSModel> model, final TimeSeries[] testSamples) {
    @SuppressWarnings("unchecked")
    final List<Pair<Double, Integer>>[] testLabels = new List[testSamples.length];
    for (int i = 0; i < testLabels.length; i++) {
      testLabels[i] = new ArrayList<>();
    }

    final List<Integer> usedLengths = Collections.synchronizedList(new ArrayList<>(model.size()));

    // parallel execution
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < model.size(); i++) {
          if (i % threads == id) {
            final BOSSModel score = model.get(i);
            usedLengths.add(score.windowLength);

            BOSS model = score.boss;

            // create words and BOSS boss for test samples
            int[][] wordsTest = model.createWords(testSamples);
            BagOfPattern[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);

            Predictions p = predict(bagTest, score.bag);

            for (int j = 0; j < p.labels.length; j++) {
              synchronized (testLabels[j]) {
                if (p.labels[j] != null) {
                  testLabels[j].add(new Pair<>(p.labels[j], score.score.training));
                }
              }
            }
          }
        }
      }
    });

    return score("BOSS", testSamples, testLabels, usedLengths);
  }

}