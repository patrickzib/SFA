// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.io.FileNotFoundException;
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

import com.carrotsearch.hppc.IntFloatOpenHashMap;
import com.carrotsearch.hppc.ObjectObjectOpenHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.ObjectObjectCursor;

/**
 *  The Bag-of-SFA-Symbols in Vector Space classifier as published in
 *    Schäfer, P.: Scalable time series classification. DMKD (Preprint)
 *    
 *    
 * @author bzcschae
 *
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

    public ObjectObjectOpenHashMap<String, E> idf;
    public BOSSVSModel model;    
    public int features;

    public void clear() {
      this.idf = null;
      this.model = null;
      
      this.testing = 0;
      this.training = 0;
    }
  }


  public Score eval() throws IOException {
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    try {
      // BOSS Distance
      BossVSScore<IntFloatOpenHashMap> totalBestScore = null;
      int bestCorrectTesting = 0;
      int bestCorrectTraining = 0;

      // generate test train/split for cross-validation
      generateIndices();

      for (boolean normMean : NORMALIZATION) {
        long startTime = System.currentTimeMillis();

        this.correctTraining = new AtomicInteger(0);

        List<BossVSScore<IntFloatOpenHashMap>> scores = fitEnsemble(exec, normMean);

        // training score
        BossVSScore<IntFloatOpenHashMap> bestScore = scores.get(0);
        if (DEBUG) {
          System.out.println("BOSS VS Training:\t" + bestScore.windowLength + " " + bestScore.features + "\tnormed: \t" + normMean);
          outputResult(this.correctTraining.get(), startTime, this.trainSamples.length);
        }

        // determine labels based on the majority of predictions
        int correctTesting = predictEnsamble(scores, this.testSamples, normMean);

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
          1-formatError(bestCorrectTesting, this.testSamples.length),
          1-formatError(bestCorrectTraining, this.trainSamples.length),
          totalBestScore.normed,
          totalBestScore.windowLength);
    }
    finally {
      exec.shutdown();
    }

  }

  public List<BossVSScore<IntFloatOpenHashMap>> fitEnsemble(
      ExecutorService exec,
      final boolean normMean) throws FileNotFoundException {
    int minWindowLength = 10;
    int maxWindowLength = this.trainSamples[0].getLength();
    for (TimeSeries ts : this.trainSamples) {
      maxWindowLength = Math.min(ts.getLength(), maxWindowLength);
    }
    maxWindowLength = Math.min(MAX_WINDOW_LENGTH, maxWindowLength);

    // equi-distance sampling of windows
    ArrayList<Integer> windows = new ArrayList<Integer>();
    double count = Math.sqrt(maxWindowLength);
    double distance = ((maxWindowLength-minWindowLength)/count);
    for (int c = minWindowLength; c <= maxWindowLength; c += distance) {
      windows.add(c);
    }

    List<BossVSScore<IntFloatOpenHashMap>> results = fit(windows.toArray(new Integer[]{}), normMean, trainSamples, exec);

    // cleanup unused scores
    for (BossVSScore<IntFloatOpenHashMap> s : results) {
      if (s.model != null
          && s.training < this.correctTraining.get() * factor) { 
        s.clear();
      }
    }

    // sort descending
    Collections.sort(results, Collections.reverseOrder());
    return results;
  }

  public List<BossVSScore<IntFloatOpenHashMap>> fit(
      Integer[] allWindows,
      boolean normMean,
      TimeSeries[] samples,
      ExecutorService exec) {
    final List<BossVSScore<IntFloatOpenHashMap>> results = new ArrayList<BossVSScore<IntFloatOpenHashMap>>(allWindows.length);
    ParallelFor.withIndex(exec, BLOCKS, new ParallelFor.Each() {
      int[] allIndices = createIndices(samples.length);
      HashSet<String> uniqueLabels = uniqueClassLabels(samples);
      BossVSScore<IntFloatOpenHashMap> bestScore = new BossVSScore<IntFloatOpenHashMap>(normMean, 0);
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < allWindows.length; i++) {
          if (i % BLOCKS == id) {
            BossVSScore<IntFloatOpenHashMap> score = new BossVSScore<IntFloatOpenHashMap>(normMean, allWindows[i]);
            try {
              BOSSVSModel model = new BOSSVSModel(maxF, maxS, score.windowLength, score.normed);              
              int[][] words = model.createWords(trainSamples);

              optimize :
                for (int f = minF; f <= Math.min(score.windowLength,maxF); f+=2) {
                  BagOfPattern[] bag = model.createBagOfPattern(words, trainSamples, f);

                  // cross validation using folds
                  int correct = 0;
                  for (int s = 0; s < folds; s++) {
                    // calculate the tf-idf for each class
                     ObjectObjectOpenHashMap<String, IntFloatOpenHashMap> idf = model.createTfIdf(bag, trainIndices[s], uniqueLabels);

                    correct += predict(testIndices[s], bag, idf).correct.get();
                  }
                  if (correct > score.training) {
                    score.training = correct;
                    score.testing = correct;
                    score.features = f;

                    if (correct == samples.length) {
                      break optimize;
                    }
                  }
                }

              // obtain the final matrix              
              BagOfPattern[] bag = model.createBagOfPattern(words, trainSamples, score.features);

              // calculate the tf-idf for each class
              score.idf = model.createTfIdf(bag, allIndices, uniqueLabels);
              score.model = model;      

            } catch (Exception e) {
              e.printStackTrace();
            }
            
            if (this.bestScore.compareTo(score)<0) {
              synchronized(this.bestScore) {
                if (this.bestScore.compareTo(score)<0) {
                  BOSSVSClassifier.this.correctTraining.set((int)score.training);
                  this.bestScore = score;
                }
              }
            }

//           if (DEBUG) {
//              System.out.println(BOSSVSClassifier.this.correctTraining.get());
//           }
          
            // add to ensemble
            if (score.training >= BOSSVSClassifier.this.correctTraining.get() * factor) {
              synchronized(results) {
                results.add(score);
              }
            }
          }
        }
      }
    });
        
    return results;
  }


  public Predictions predict(
      final int[] indices,
      final BagOfPattern[] bagOfPatternsTestSamples,
      final ObjectObjectOpenHashMap<String, IntFloatOpenHashMap> matrixTrain) {

    Predictions p = new Predictions(new String[bagOfPatternsTestSamples.length], 0);

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        predict(id, processed);
      }
      public void predict(int id, AtomicInteger processed) {
        // iterate each sample to classify
        for (int i : indices) {
          if (i % BLOCKS == id) {
            double bestDistance = 0.0;

            // for each class
            for (ObjectObjectCursor<String, IntFloatOpenHashMap> classEntry : matrixTrain) {

              String label = classEntry.key;
              IntFloatOpenHashMap stat = classEntry.value;

              // determine cosine similarity
              double distance = 0.0;
              for (IntIntCursor wordFreq : bagOfPatternsTestSamples[i].bag) {
                double wordInBagFreq = wordFreq.value;
                double value = stat.get(wordFreq.key);
                distance += wordInBagFreq * (value+1.0);
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

  public int predictEnsamble(
      final List<BossVSScore<IntFloatOpenHashMap>> results,
      final TimeSeries[] testSamples,
      boolean normMean) {
    ExecutorService executor = Executors.newFixedThreadPool(BLOCKS);
    long startTime = System.currentTimeMillis();

    @SuppressWarnings("unchecked")
    final List<Pair<String, Double>>[] testLabels = new List[testSamples.length];
    for (int i = 0; i < testLabels.length; i++) {
      testLabels[i] = new ArrayList<Pair<String, Double>>();
    }

    final List<Integer> usedLengths = new ArrayList<Integer>(results.size());

    try {
      final int[] indicesTest = createIndices(testSamples.length);

      // parallel execution
      ParallelFor.withIndex(executor, BLOCKS, new ParallelFor.Each() {
        @Override
        public void run(int id, AtomicInteger processed) {
          predictEnsambleLabel(id, processed);
        }
        public void predictEnsambleLabel(int id, AtomicInteger processed) {
          // iterate each sample to classify
          for (int i = 0; i < results.size(); i++) {
            if (i % BLOCKS == id) {
              final BossVSScore<IntFloatOpenHashMap> score = results.get(i);
              if (score.training >= BOSSVSClassifier.this.correctTraining.get() * factor) { // all with same score
                usedLengths.add(score.windowLength);

                BOSSVSModel model = score.model;
                
                // create words and BOSS model for test samples
                int[][] wordsTest = model.createWords(testSamples);
                BagOfPattern[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);

                Predictions p = predict(indicesTest, bagTest, score.idf);

                for (int j = 0; j < p.labels.length; j++) {
                  synchronized (testLabels[j]) {
                    testLabels[j].add(new Pair<String, Double>(p.labels[j], score.training));
                  }
                }
              }
              else {
                score.clear();
              }
            }
          }
        }
      });
    } finally {
      executor.shutdown();
    }

    return score("BOSS VS", testSamples, startTime, testLabels, usedLengths);
  }
}