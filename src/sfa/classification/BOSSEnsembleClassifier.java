package sfa.classification;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.BOSSModel;
import sfa.transformation.BOSSModel.BagOfPattern;

import com.carrotsearch.hppc.cursors.IntIntCursor;

/**
 * 
 * The Bag-of-SFA-Symbols Ensemble Classifier as published in
 *    Schäfer, P.: The boss is concerned with time series classification 
 *    in the presence of noise. DMKD 29(6) (2015) 1505–1530
 *   
 * @author bzcschae
 *
 */
public class BOSSEnsembleClassifier extends Classifier {
  
  public static double factor = 0.92;

  public static int maxF = 16; // 12
  public static int minF = 6;  // 4
  public static int maxS = 4;  // 8

  public BOSSEnsembleClassifier(TimeSeries[] train, TimeSeries[] test) throws IOException {
    super(train, test);
  }

  public static class BossScore extends Score {
    public BossScore(boolean normed, int windowLength) {
      super("BOSS", 0, 0, normed, windowLength);
    }

    public BagOfPattern[] bag;
    public BOSSModel model;    
    public int features;

    public void clear() {
      this.model = null;
      this.bag = null;
      this.testing = 0;
      this.training = 0;
    }
  }

  public Score eval() throws IOException {
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    try {
      BossScore totalBestScore = null;
      int bestCorrectTesting = 0;
      int bestCorrectTraining = 0;

      for (boolean norm : NORMALIZATION) {
        long startTime = System.currentTimeMillis();

        this.correctTraining = new AtomicInteger(0);

        List<BossScore> scores = fitEnsemble(exec, norm);

        // training score
        BossScore bestScore = scores.get(0);
        if (DEBUG) {
          System.out.println("BOSS Training:\t" + bestScore.windowLength + " " + bestScore.features + "\tnormed: \t" + norm);
          outputResult(this.correctTraining.get(), startTime, this.trainSamples.length);
        }

        // determine labels based on the majority of predictions
        int correctTesting = predictEnsamble(scores, this.testSamples, norm);

        if (bestCorrectTraining < bestScore.training) {
          bestCorrectTesting = correctTesting;
          bestCorrectTraining = (int)bestScore.training;
          totalBestScore = bestScore;
        }
        if (DEBUG) {
          System.out.println("");
        }
      }

      return new Score(
          "BOSS ensemble",
          1-formatError(bestCorrectTesting, this.testSamples.length),
          1-formatError(bestCorrectTraining, this.trainSamples.length),
          totalBestScore.normed,
          totalBestScore.windowLength);
    }
    finally {
      exec.shutdown();
    }
  }

  public List<BossScore> fitEnsemble(
      ExecutorService exec,
      final boolean normMean) throws FileNotFoundException {
    int minWindowLength = 10;
    int maxWindowLength = this.trainSamples[0].getLength();
    for (TimeSeries ts : this.trainSamples) {
      maxWindowLength = Math.min(ts.getLength(), maxWindowLength);
    }
    maxWindowLength = Math.min(MAX_WINDOW_LENGTH, maxWindowLength);

    ArrayList<Integer> windows = new ArrayList<Integer>();
    for (int windowLength = maxWindowLength; windowLength >= minWindowLength; windowLength--) {
      windows.add(windowLength);
    }
//    windows.add(15);

    List<BossScore> results = fit(windows.toArray(new Integer[]{}), normMean, trainSamples, exec);
    
    // cleanup unused scores
    for (BossScore s : results) {
      if (s.bag != null
          && s.training < BOSSEnsembleClassifier.this.correctTraining.get() * factor) { 
        s.clear();
      }
    }

    // sort descending
    Collections.sort(results, Collections.reverseOrder());
    return results;
  }

  public ArrayList<BossScore> fit(
      Integer[] allWindows,
      boolean normMean,
      TimeSeries[] samples,
      ExecutorService exec) {
    final ArrayList<BossScore> results = new ArrayList<BossScore>(allWindows.length);
    ParallelFor.withIndex(exec, BLOCKS, new ParallelFor.Each() {
      BossScore bestScore = new BossScore(normMean, 0);
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < allWindows.length; i++) {
          if (i % BLOCKS == id) {         
            BossScore score = new BossScore(normMean, allWindows[i]);
            try {
              BOSSModel boss = new BOSSModel(maxF, maxS, score.windowLength, score.normed);              
              int[][] words = boss.createWords(samples);
              
              optimize :
                for (int f = minF; f <= maxF; f+=2) {
                  
                  BagOfPattern[] bag = boss.createBagOfPattern(words, samples, f);
                  
                  Predictions p = predict(score.windowLength, bag, bag);

                  if (p.correct.get() > score.training) {
                    score.training = p.correct.get();
                    score.testing = p.correct.get();
                    score.features = f;
                    score.model = boss;
                    score.bag = bag;
                    
                    if (p.correct.get() == samples.length) {
                      break optimize;
                    }
                  }
                }
            } catch (Exception e) {
              e.printStackTrace();
            }
            
            // keep best scores
            if (this.bestScore.compareTo(score)<0) {
              synchronized(this.bestScore) {
                if (this.bestScore.compareTo(score)<0) {
                  BOSSEnsembleClassifier.this.correctTraining.set((int)score.training);
                  this.bestScore = score;
                }
              }              
            }

//            if (DEBUG) {
//              System.out.println(i + "\t" + currentScore.training);
//            }
            
            // add to ensemble
            if (score.training >= BOSSEnsembleClassifier.this.correctTraining.get() * factor) { // all with same score
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
      final int windowLength,
      final BagOfPattern[] bagOfPatternsTestSamples,
      final BagOfPattern[] bagOfPatternsTrainSamples) {

    Predictions p = new Predictions(new String[bagOfPatternsTestSamples.length], 0);

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        predict(id, processed);
      }
      public void predict(int id, AtomicInteger processed) {
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
            p.labels[i] = bestMatch > -1? bagOfPatternsTrainSamples[bestMatch].label : null;
            if (bagOfPatternsTestSamples[i].label.equals(p.labels[i])) {
              p.correct.incrementAndGet();
            }
//            System.out.println(i + " " + Double.valueOf(label[i]).intValue());
          }
        }
      }
    });

    return p;
  }

  public int predictEnsamble(
      final List<BossScore> results,
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
              final BossScore score = results.get(i);
              if (score.training >= BOSSEnsembleClassifier.this.correctTraining.get() * factor) { // all with same score
                usedLengths.add(score.windowLength);                

                BOSSModel model = score.model;
                
                // create words and BOSS model for test samples
                int[][] wordsTest = model.createWords(testSamples);                
                BagOfPattern[] bagTest = model.createBagOfPattern(wordsTest, testSamples, score.features);
                
                Predictions p = predict(score.windowLength, bagTest, score.bag);

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

    return score("BOSS", testSamples, startTime, testLabels, usedLengths);
  }

}