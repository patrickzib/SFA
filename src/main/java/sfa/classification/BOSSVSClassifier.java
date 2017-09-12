// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.BOSSModel.BagOfPattern;
import sfa.transformation.BOSSVSModel;

import com.carrotsearch.hppc.IntFloatHashMap;
import com.carrotsearch.hppc.ObjectObjectHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.ObjectObjectCursor;

/**
 * The Bag-of-SFA-Symbols in Vector Space classifier as published in
 * <p>
 * Schäfer, P.: Scalable time series classification. DMKD (2016)
 */
public class BOSSVSClassifier extends Classifier {

  public static double factor = 0.95;

  public static int maxF = 16;
  public static int minF = 4;
  public static int maxS = 4;

  public static boolean normMagnitudes = false;

  public BOSSVSClassifier(TimeSeries[] train, TimeSeries[] test) throws IOException {
    super(train, test);
  }

  public static class BossVSScore<E> extends Score {
    public BossVSScore(boolean normed, int windowLength) {
      super("BOSS VS", 0, 0, normed, windowLength);
    }

    public ObjectObjectHashMap<String, E> idf;
    public BOSSVSModel model;
    public int features;

    @Override
    public void clear() {
      super.clear();
      this.idf = null;
      this.model = null;
    }
  }


  @Override
  public Score eval() throws IOException {
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    try {
      // BOSS Distance
      BossVSScore<IntFloatHashMap> totalBestScore = null;
      int bestCorrectTesting = 0;
      int bestCorrectTraining = 0;

      // generate test train/split for cross-validation
      generateIndices();

      for (boolean normMean : NORMALIZATION) {
        long startTime = System.currentTimeMillis();

        this.correctTraining = new AtomicInteger(0);

        List<BossVSScore<IntFloatHashMap>> scores = fitEnsemble(exec, normMean);

        // training score
        BossVSScore<IntFloatHashMap> bestScore = scores.get(0);
        if (DEBUG) {
          System.out.println("BOSS VS Training:\t" + bestScore.windowLength + " " + bestScore.features + "\tnormed: \t" + normMean);
          outputResult(this.correctTraining.get(), startTime, this.trainSamples.length);
        }

        // determine labels based on the majority of predictions
        int correctTesting = predictEnsemble(exec, scores, this.testSamples, normMean);

        if (bestCorrectTraining <= this.correctTraining.get()) {
          bestCorrectTesting = correctTesting;
          bestCorrectTraining = this.correctTraining.get();
          totalBestScore = bestScore;
        }
        if (DEBUG) {
          System.out.println("");
        }
      }

      return new Score(
          "BOSS VS",
          1 - formatError(bestCorrectTesting, this.testSamples.length),
          1 - formatError(bestCorrectTraining, this.trainSamples.length),
          totalBestScore.normed,
          totalBestScore.windowLength);
    } finally {
      exec.shutdown();
    }

  }

  public List<BossVSScore<IntFloatHashMap>> fitEnsemble(ExecutorService exec, final boolean normMean) {
    int minWindowLength = 10;
    int maxWindowLength = getMax(this.trainSamples, MAX_WINDOW_LENGTH);

    // equi-distance sampling of windows
    ArrayList<Integer> windows = new ArrayList<>();
    double count = Math.sqrt(maxWindowLength);
    double distance = ((maxWindowLength - minWindowLength) / count);
    for (int c = minWindowLength; c <= maxWindowLength; c += distance) {
      windows.add(c);
    }
    return fit(windows.toArray(new Integer[]{}), normMean, this.trainSamples, exec);
  }

  public List<BossVSScore<IntFloatHashMap>> fit(
      Integer[] allWindows,
      boolean normMean,
      TimeSeries[] samples,
      ExecutorService exec) {
    final List<BossVSScore<IntFloatHashMap>> results = new ArrayList<>(allWindows.length);
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      HashSet<String> uniqueLabels = uniqueClassLabels(samples);
      BossVSScore<IntFloatHashMap> bestScore = new BossVSScore<>(normMean, 0);
      final Object sync = new Object();

      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < allWindows.length; i++) {
          if (i % threads == id) {
            BossVSScore<IntFloatHashMap> score = new BossVSScore<>(normMean, allWindows[i]);
            try {
              BOSSVSModel model = new BOSSVSModel(maxF, maxS, score.windowLength, score.normed);
              int[][] words = model.createWords(BOSSVSClassifier.this.trainSamples);

              for (int f = minF; f <= Math.min(score.windowLength, maxF); f += 2) {
                BagOfPattern[] bag = model.createBagOfPattern(words, BOSSVSClassifier.this.trainSamples, f);

                // cross validation using folds
                int correct = 0;
                for (int s = 0; s < folds; s++) {
                  // calculate the tf-idf for each class
                  ObjectObjectHashMap<String, IntFloatHashMap> idf = model.createTfIdf(bag,
                      BOSSVSClassifier.this.trainIndices[s], this.uniqueLabels);

                  correct += predict(BOSSVSClassifier.this.testIndices[s], bag, idf).correct.get();
                }
                if (correct > score.training) {
                  score.training = correct;
                  score.testing = correct;
                  score.features = f;

                  if (correct == samples.length) {
                    break;
                  }
                }
              }

              // obtain the final matrix
              BagOfPattern[] bag = model.createBagOfPattern(words, BOSSVSClassifier.this.trainSamples, score.features);

              // calculate the tf-idf for each class
              score.idf = model.createTfIdf(bag, this.uniqueLabels);
              score.model = model;

            } catch (Exception e) {
              e.printStackTrace();
            }

            synchronized (sync) {
              if (this.bestScore.compareTo(score) < 0) {
                BOSSVSClassifier.this.correctTraining.set((int) score.training);
                this.bestScore = score;
              }
            }

            // add to ensemble
            if (score.training >= BOSSVSClassifier.this.correctTraining.get() * factor) {
              synchronized (results) {
                results.add(score);
              }
            }
          }
        }
      }
    });

    // cleanup unused scores
    for (BossVSScore<IntFloatHashMap> s : results) {
      if (s.model != null
          && s.training < this.correctTraining.get() * factor) {
        s.clear();
      }
    }

    // sort descending
    Collections.sort(results, Collections.reverseOrder());
    return results;
  }


  public Predictions predict(
      final int[] indices,
      final BagOfPattern[] bagOfPatternsTestSamples,
      final ObjectObjectHashMap<String, IntFloatHashMap> matrixTrain) {

    Predictions p = new Predictions(new String[bagOfPatternsTestSamples.length], 0);

    ParallelFor.withIndex(this.BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i : indices) {
          if (i % BOSSVSClassifier.this.BLOCKS == id) {
            double bestDistance = 0.0;

            // for each class
            for (ObjectObjectCursor<String, IntFloatHashMap> classEntry : matrixTrain) {

              String label = classEntry.key;
              IntFloatHashMap stat = classEntry.value;

              // determine cosine similarity
              double distance = 0.0;
              for (IntIntCursor wordFreq : bagOfPatternsTestSamples[i].bag) {
                double wordInBagFreq = wordFreq.value;
                double value = stat.get(wordFreq.key);
                distance += wordInBagFreq * (value + 1.0);
              }

              // norm by magnitudes
              if (normMagnitudes) {
                distance /= magnitude(stat.values());
              }

              // update nearest neighbor
              if (distance > bestDistance) {
                bestDistance = distance;
                p.labels[i] = label;
              }
            }

            // check if the prediction is correct
            if (bagOfPatternsTestSamples[i].label.equals(p.labels[i])) {
              p.correct.incrementAndGet();
            }
          }
        }
      }
    });

    return p;
  }

  public int predictEnsemble(
      ExecutorService executor,
      final List<BossVSScore<IntFloatHashMap>> results,
      final TimeSeries[] testSamples,
      boolean normMean) {
    long startTime = System.currentTimeMillis();

    @SuppressWarnings("unchecked")
    final List<Pair<String, Double>>[] testLabels = new List[testSamples.length];
    for (int i = 0; i < testLabels.length; i++) {
      testLabels[i] = new ArrayList<>();
    }

    final List<Integer> usedLengths = new ArrayList<>(results.size());
    final int[] indicesTest = createIndices(testSamples.length);

    // parallel execution
    ParallelFor.withIndex(executor, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < results.size(); i++) {
          if (i % threads == id) {
            final BossVSScore<IntFloatHashMap> score = results.get(i);
            if (score.training >= BOSSVSClassifier.this.correctTraining.get() * factor) { // all with same score
              usedLengths.add(score.windowLength);

              BOSSVSModel model = score.model;

              // create words and BOSS model for test samples
              int[][] wordsTest = model.createWords(testSamples);
              BagOfPattern[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);

              Predictions p = predict(indicesTest, bagTest, score.idf);

              for (int j = 0; j < p.labels.length; j++) {
                synchronized (testLabels[j]) {
                  testLabels[j].add(new Pair<>(p.labels[j], score.training));
                }
              }
            } else {
              score.clear();
            }
          }
        }
      }
    });

    return score("BOSS VS", testSamples, startTime, testLabels, usedLengths);
  }
}