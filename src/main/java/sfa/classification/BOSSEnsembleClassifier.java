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

  // the trained weasel
  public Ensemble<BOSSModel> model;

  public BOSSEnsembleClassifier(TimeSeries[] train, TimeSeries[] test) {
    super(train, test);
  }

  public static class BOSSModel extends Model {
    public BOSSModel(
        boolean normed,
        int windowLength) {
      super("BOSS", -1, -1, normed, windowLength);
    }

    public BagOfPattern[] bag;
    public BOSS boss;
    public int features;
  }

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
        "BOSS Ensemble",
        1 - formatError(correctTesting, this.testSamples.length),
        1 - formatError((int) score.training, this.trainSamples.length),
        score.windowLength);
  }


  public Score fit(final TimeSeries[] samples) {
    // generate test train/split for cross-validation
    generateIndices();

    Score bestScore = null;
    int bestCorrectTraining = 0;

    for (boolean normMean : NORMALIZATION) {
      // train the shotgun models for different window lengths
      Ensemble<BOSSModel> model = fitEnsemble(samples, normMean);
      Score score = model.getHighestScoringModel().score;

      if (bestCorrectTraining < score.training) {
        bestCorrectTraining = (int) score.training;
        bestScore = score;
        this.model = model;
      }
    }

    // return score
    return bestScore;
  }


  protected Ensemble<BOSSModel> fitEnsemble(
      final TimeSeries[] samples,
      final boolean normMean) {
    int minWindowLength = 10;
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

    final ArrayList<BOSSModel> results = new ArrayList<>(allWindows.length);
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      Score bestScore = new Score("BOSS", 0, 0, 0);
      final Object sync = new Object();

      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < allWindows.length; i++) {
          if (i % threads == id) {
            BOSSModel model = new BOSSModel(normMean, allWindows[i]);
            try {
              BOSS boss = new BOSS(maxF, maxS, allWindows[i], model.normed);
              int[][] words = boss.createWords(samples);

              for (int f = minF; f <= maxF; f += 2) {

                BagOfPattern[] bag = boss.createBagOfPattern(words, samples, f);

                Predictions p = predict(bag, bag);

                if (p.correct.get() > model.score.training) {
                  model.score.training = p.correct.get();
                  model.score.testing = p.correct.get();
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
            synchronized (sync) {
              if (this.bestScore.compareTo(model.score) < 0) {
                correctTraining.set((int) model.score.training);
                this.bestScore = model.score;
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
    List<BOSSModel> model = new ArrayList<>();

    for (int i = 0; i < results.size(); i++) {
      final BOSSModel score = results.get(i);
      if (score.score.training >= correctTraining.get() * factor) { // all with same score
        model.add(score);
      }
    }

    return new Ensemble<>(results);
  }


  public Predictions predict(final TimeSeries[] testSamples) {
    return predictEnsemble(this.model, testSamples);
  }

  protected Predictions predict(
      final BagOfPattern[] bagOfPatternsTestSamples,
      final BagOfPattern[] bagOfPatternsTrainSamples) {

    Predictions p = new Predictions(new String[bagOfPatternsTestSamples.length], 0);

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
          if (i % BLOCKS == id) {
            int bestMatch = -1;
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
                  bestMatch = j;
                }
              }
            }

            // check if the prediction is correct
            p.labels[i] = bestMatch > -1 ? bagOfPatternsTrainSamples[bestMatch].label : null;
            if (bagOfPatternsTestSamples[i].label.equals(p.labels[i])) {
              p.correct.incrementAndGet();
            }
          }
        }
      }
    });

    return p;
  }

  protected Predictions predictEnsemble(
      final Ensemble<BOSSModel> results,
      final TimeSeries[] testSamples) {
    long startTime = System.currentTimeMillis();

    @SuppressWarnings("unchecked")
    final List<Pair<String, Double>>[] testLabels = new List[testSamples.length];
    for (int i = 0; i < testLabels.length; i++) {
      testLabels[i] = new ArrayList<>();
    }

    final List<Integer> usedLengths = new ArrayList<>(results.size());

    // parallel execution
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < results.size(); i++) {
          if (i % threads == id) {
            final BOSSModel score = results.get(i);
            usedLengths.add(score.windowLength);

            BOSS model = score.boss;

            // create words and BOSS boss for test samples
            int[][] wordsTest = model.createWords(testSamples);
            BagOfPattern[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);

            Predictions p = predict(bagTest, score.bag);

            for (int j = 0; j < p.labels.length; j++) {
              synchronized (testLabels[j]) {
                testLabels[j].add(new Pair<>(p.labels[j], score.score.training));
              }
            }
          }
        }
      }
    });

    return score("BOSS", testSamples, startTime, testLabels, usedLengths);
  }

}