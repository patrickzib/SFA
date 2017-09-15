// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.BOSS.BagOfPattern;
import sfa.transformation.BOSSVS;

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

  // default training parameters
  public static double factor = 0.95;

  public static int maxF = 16;
  public static int minF = 4;
  public static int maxS = 4;

  public static boolean normMagnitudes = false;

  // the trained weasel
  public Ensemble<BossVSModel<IntFloatHashMap>> model;

  public BOSSVSClassifier() {
    super();
  }


  public static class BossVSModel<E> extends Model {
    public BossVSModel(
        boolean normed,
        int windowLength) {
      super("BOSS VS", -1, 1, -1, 1, normed, windowLength);
    }

    public ObjectObjectHashMap<String, E> idf;
    public BOSSVS bossvs;
    public int features;
  }


  @Override
  public Score eval(
      final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
    long startTime = System.currentTimeMillis();

    Score score = fit(trainSamples);

    if (DEBUG) {
      System.out.println(score.toString());
      outputResult((int) score.training, startTime, testSamples.length);
      System.out.println("");
    }

    // Classify: testing score
    int correctTesting = predict(testSamples).correct.get();

    return new Score(
        "BOSS VS",
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

    for (boolean normMean : NORMALIZATION) {
      // train the shotgun models for different window lengths
      Ensemble<BossVSModel<IntFloatHashMap>> model = fitEnsemble(trainSamples, normMean);
      Score score = model.getHighestScoringModel().score;

      Predictions pred = predictEnsemble(model, trainSamples);

      if (bestCorrectTraining <= pred.correct.get()) {
        bestCorrectTraining = pred.correct.get();
        bestScore = score;
        bestScore.training = pred.correct.get();
        this.model = model;
      }
    }

    // return score
    return bestScore;
  }


  @Override
  public Predictions predict(final TimeSeries[] testSamples) {
    return predictEnsemble(this.model, testSamples);
  }


  protected Ensemble<BossVSModel<IntFloatHashMap>> fitEnsemble(
      final TimeSeries[] samples,
      final boolean normMean) {
    int minWindowLength = 10;
    int maxWindowLength = getMax(samples, MAX_WINDOW_LENGTH);

    // equi-distance sampling of windows
    ArrayList<Integer> windows = new ArrayList<>();
    double count = Math.sqrt(maxWindowLength);
    double distance = ((maxWindowLength - minWindowLength) / count);
    for (int c = minWindowLength; c <= maxWindowLength; c += distance) {
      windows.add(c);
    }
    return fit(windows.toArray(new Integer[]{}), normMean, samples);
  }


  protected Ensemble<BossVSModel<IntFloatHashMap>> fit(
      Integer[] allWindows,
      boolean normMean,
      TimeSeries[] samples) {

    final List<BossVSModel<IntFloatHashMap>> results = new ArrayList<>(allWindows.length);

    final AtomicInteger correctTraining = new AtomicInteger(0);

    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      HashSet<String> uniqueLabels = uniqueClassLabels(samples);
      Score bestScore = new Score("BOSS VS", 0, 1, 0, 1, 0);
      final Object sync = new Object();

      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < allWindows.length; i++) {
          if (i % threads == id) {
            BossVSModel<IntFloatHashMap> model = new BossVSModel<>(normMean, allWindows[i]);
            try {
              BOSSVS bossvs = new BOSSVS(maxF, maxS, allWindows[i], model.normed);
              int[][] words = bossvs.createWords(samples);

              for (int f = minF; f <= Math.min(model.windowLength, maxF); f += 2) {
                BagOfPattern[] bag = bossvs.createBagOfPattern(words, samples, f);

                // cross validation using folds
                int correct = 0;
                for (int s = 0; s < folds; s++) {
                  // calculate the tf-idf for each class
                  ObjectObjectHashMap<String, IntFloatHashMap> idf = bossvs.createTfIdf(bag,
                      BOSSVSClassifier.this.trainIndices[s], this.uniqueLabels);

                  correct += predict(BOSSVSClassifier.this.testIndices[s], bag, idf).correct.get();
                }
                if (correct > model.score.training) {
                  model.score.training = correct;
                  model.score.testing = correct;
                  model.features = f;

                  if (correct == samples.length) {
                    break;
                  }
                }
              }

              // obtain the final matrix
              BagOfPattern[] bag = bossvs.createBagOfPattern(words, samples, model.features);

              // calculate the tf-idf for each class
              model.idf = bossvs.createTfIdf(bag, this.uniqueLabels);
              model.bossvs = bossvs;

            } catch (Exception e) {
              e.printStackTrace();
            }

            synchronized (sync) {
              if (this.bestScore.compareTo(model.score) < 0) {
                correctTraining.set((int) model.score.training);
                this.bestScore = model.score;
              }
            }

            // add to ensemble
            if (model.score.training >= correctTraining.get() * factor) {
              synchronized (results) {
                results.add(model);
              }
            }
          }
        }
      }
    });

    // only keep best scores
    List<BossVSModel<IntFloatHashMap>> model = new ArrayList<>();

    for (int i = 0; i < results.size(); i++) {
      final BossVSModel<IntFloatHashMap> score = results.get(i);
      if (score.score.training >= correctTraining.get() * factor) { // all with same score
        model.add(score);
      }
    }

    return new Ensemble<>(model);
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


  protected Predictions predictEnsemble(
      final Ensemble<BossVSModel<IntFloatHashMap>> results,
      final TimeSeries[] testSamples) {
    long startTime = System.currentTimeMillis();

    @SuppressWarnings("unchecked")
    final List<Pair<String, Integer>>[] testLabels = new List[testSamples.length];
    for (int i = 0; i < testLabels.length; i++) {
      testLabels[i] = new ArrayList<>();
    }

    final List<Integer> usedLengths = new ArrayList<>(results.size());
    final int[] indicesTest = createIndices(testSamples.length);

    // parallel execution
    ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i = 0; i < results.size(); i++) {
          if (i % threads == id) {
            final BossVSModel<IntFloatHashMap> score = results.get(i);
            usedLengths.add(score.windowLength);

            BOSSVS model = score.bossvs;

            // create words and BOSS boss for test samples
            int[][] wordsTest = model.createWords(testSamples);
            BagOfPattern[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);

            Predictions p = predict(indicesTest, bagTest, score.idf);

            for (int j = 0; j < p.labels.length; j++) {
              synchronized (testLabels[j]) {
                testLabels[j].add(new Pair<>(p.labels[j], score.score.training));
              }
            }
          }
        }
      }
    });

    return score("BOSS VS", testSamples, startTime, testLabels, usedLengths);
  }
}