// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import com.carrotsearch.hppc.cursors.LongIntCursor;
import de.bwaldvogel.liblinear.*;
import sfa.timeseries.TimeSeries;
import sfa.transformation.WEASEL;
import sfa.transformation.WEASEL.BagOfBigrams;
import sfa.transformation.WEASEL.Dictionary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The WEASEL (Word ExtrAction for time SEries cLassification) classifier as published in
 * <p>
 * Schäfer, P., Leser, U.: Fast and Accurate Time Series
 * Classification with WEASEL." CIKM 2017
 */
public class WEASELClassifier extends Classifier {

  // default training parameters
  public static int maxF = 6;
  public static int minF = 4;
  public static int maxS = 4;

  public static SolverType solverType = SolverType.L2R_LR_DUAL;

  public static double chi = 2;
  public static double bias = 1;
  public static double eps = 0.1;
  public static int iterations = 5000;
  public static double c = 1;

  public static boolean lowerBounding = false;

  public static int MIN_WINDOW_LENGTH = 2;
  public static int MAX_WINDOW_LENGTH = 350;

  // the trained weasel
  WEASELModel model;

  public WEASELClassifier() {
    super();
    Linear.resetRandom();
    Linear.disableDebugOutput();
  }

  public static class WEASELModel extends Model {

    public WEASELModel(){}

    public WEASELModel(
        boolean normed,
        int features,
        WEASEL model,
        de.bwaldvogel.liblinear.Model linearModel,
        int testing,
        int testSize,
        int training,
        int trainSize
    ) {
      super("WEASEL", testing, testSize, training, trainSize, normed, -1);
      this.features = features;
      this.weasel = model;
      this.linearModel = linearModel;
    }

    // the best number of Fourier values to be used
    public int features;

    // the trained WEASEL transformation
    public WEASEL weasel;

    // the trained liblinear classifier
    public de.bwaldvogel.liblinear.Model linearModel;
  }

  @Override
  public Score eval(
      final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
    long startTime = System.currentTimeMillis();

    Score score = fit(trainSamples);

    // training score
    if (DEBUG) {
      System.out.println(score.toString());
      outputResult(score.training, startTime, trainSamples.length);
    }

    // determine score
    int correctTesting = score(testSamples).correct.get();

    if (DEBUG) {
      System.out.println("WEASEL Testing:\t");
      outputResult(correctTesting, startTime, testSamples.length);
      System.out.println("");
    }

    return new Score(
        "WEASEL",
        correctTesting, testSamples.length,
        score.training, trainSamples.length,
        score.windowLength
    );
  }


  @Override
  public Score fit(final TimeSeries[] trainSamples) {
    // train the shotgun models for different window lengths
    this.model = fitWeasel(trainSamples);

    // return score
    return model.score;
  }


  @Override
  public Predictions score(final TimeSeries[] testSamples) {
    Double[] labels = predict(testSamples);
    return evalLabels(testSamples, labels);
  }

  public Double[] predict(TimeSeries[] samples) {

    // iterate each sample to classify
    final BagOfBigrams[] bagTest = new BagOfBigrams[samples.length];
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < model.weasel.windowLengths.length; w++) {
          if (w % BLOCKS == id) {
            int[][] wordsTest = model.weasel.createWords(samples, w);
            BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(wordsTest, samples, w, model.features);
            model.weasel.dict.filterChiSquared(bopForWindow);
            mergeBobs(bagTest, bopForWindow);
          }
        }
      }
    });

    FeatureNode[][] features = initLibLinear(bagTest, model.weasel.dict);
    Double[] labels = new Double[samples.length];

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int ind = 0; ind < features.length; ind++) {
          if (ind % BLOCKS == id) {
            double label = Linear.predict(model.linearModel, features[ind]);
            labels[ind] = label;
          }
        }
      }
    });

    return labels;
  }

  public Predictions predictProbabilities(TimeSeries[] samples) {
    final Double[] labels = new Double[samples.length];
    final double[][] probabilities = new double[samples.length][];

    // iterate each sample to classify
    final BagOfBigrams[] bagTest = new BagOfBigrams[samples.length];
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < model.weasel.windowLengths.length; w++) {
          if (w % BLOCKS == id) {
            int[][] wordsTest = model.weasel.createWords(samples, w);
            BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(wordsTest, samples, w, model.features);
            model.weasel.dict.filterChiSquared(bopForWindow);
            mergeBobs(bagTest, bopForWindow);
          }
        }
      }
    });

    FeatureNode[][] features = initLibLinear(bagTest, model.weasel.dict);

    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int ind = 0; ind < features.length; ind++) {
          if (ind % BLOCKS == id) {
            probabilities[ind] = new double[model.linearModel.getNrClass()];
            labels[ind] = Linear.predictProbability(model.linearModel, features[ind], probabilities[ind]);
          }
        }
      }
    });

    return new Predictions(labels, probabilities, model.linearModel.getLabels());
  }

  public int[] getWindowLengths(final TimeSeries[] samples, boolean norm) {
    int min = norm && MIN_WINDOW_LENGTH<=2? Math.max(3,MIN_WINDOW_LENGTH) : MIN_WINDOW_LENGTH;
    int max = getMax(samples, MAX_WINDOW_LENGTH);

    int[] wLengths = new int[max - min + 1];
    int a = 0;
    for (int w = min; w <= max; w+=1, a++) {
      wLengths[a] = w;
    }
    return Arrays.copyOfRange(wLengths, 0, a);
  }

  protected WEASELModel fitWeasel(final TimeSeries[] samples) {
    try {
      int maxCorrect = -1;
      int bestF = -1;
      boolean bestNorm = false;

      optimize:
      for (final boolean mean : NORMALIZATION) {
        int[] windowLengths = getWindowLengths(samples, mean);
        WEASEL model = new WEASEL(maxF, maxS, windowLengths, mean, lowerBounding);
        final int[][][] words = model.createWords(samples);

        for (int f = minF; f <= maxF; f += 2) {
          model.dict.reset();

          final BagOfBigrams[] bop = new BagOfBigrams[samples.length];
          final int ff = f;

          ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
              for (int w = 0; w < model.windowLengths.length; w++) {
                if (w % BLOCKS == id) {
                  BagOfBigrams[] bobForOneWindow = fitOneWindow(
                      samples,
                      model.windowLengths, mean,
                      words[w], ff, w);
                  mergeBobs(bop, bobForOneWindow);
                }
              }
            }
          });

          // train liblinear
          final Problem problem = initLibLinearProblem(bop, model.dict, bias);

          if (DEBUG)
            System.out.println("Train Dict Size: " + model.dict.size() + " Memory: " + getUsedMemory() + " MB");
          int correct = trainLibLinear(problem, solverType, c, iterations, eps, folds);

          if (correct > maxCorrect) {
            maxCorrect = correct;
            bestF = f;
            bestNorm = mean;
          }
          if (correct == samples.length) {
            break optimize;
          }
        }
      }

      // obtain the final matrix
      int[] windowLengths = getWindowLengths(samples, bestNorm);
      WEASEL model = new WEASEL(maxF, maxS, windowLengths, bestNorm, lowerBounding);

      final BagOfBigrams[] bop = new BagOfBigrams[samples.length];
      final boolean mean = bestNorm;
      final int ff = bestF;
      ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
        @Override
        public void run(int id, AtomicInteger processed) {
          for (int w = 0; w < model.windowLengths.length; w++) {
            if (w % BLOCKS == id) {
              int[][] words = model.createWords(samples, w);
              BagOfBigrams[] bobForOneWindow = fitOneWindow(
                  samples,
                  model.windowLengths, mean,
                  words, ff, w);
              mergeBobs(bop, bobForOneWindow);
            }
          }
        }
      });

      // train liblinear
      Problem problem = initLibLinearProblem(bop, model.dict, bias);
      if (DEBUG)
        System.out.println("Final Dict Size: " + model.dict.size() + " Memory: " + getUsedMemory() + " MB");
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, eps));

      return new WEASELModel(
          bestNorm,
          bestF,
          model,
          linearModel,
          0, // testing
          1,
          maxCorrect, // training
          samples.length
      );

    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  private BagOfBigrams[] fitOneWindow(
      TimeSeries[] samples,
      int[] windowLengths,
      boolean mean,
      int[][] word, int f,
      int w) {
    WEASEL modelForWindow = new WEASEL(f, maxS, windowLengths, mean, lowerBounding);
    BagOfBigrams[] bopForWindow = modelForWindow.createBagOfPatterns(word, samples, w, f);
    modelForWindow.trainChiSquared(bopForWindow, chi);
    //modelForWindow.trainAnova(bopForWindow, p_value);
    return bopForWindow;
  }

  private synchronized void mergeBobs(
      BagOfBigrams[] bop,
      BagOfBigrams[] bopForWindow) {
    for (int i = 0; i < bop.length; i++) {
      if (bop[i]==null) {
        bop[i] = bopForWindow[i];
      }
      else {
        bop[i].bob.putAll(bopForWindow[i].bob);
      }
    }
  }

  protected static Problem initLibLinearProblem(
      final BagOfBigrams[] bob,
      final Dictionary dict,
      final double bias) {
    Linear.resetRandom();
    Linear.disableDebugOutput();

    Problem problem = new Problem();
    problem.bias = bias;
    problem.y = getLabels(bob);

    final FeatureNode[][] features = initLibLinear(bob, dict);

    problem.n = dict.size() + 1;
    problem.l = features.length;
    problem.x = features;
    return problem;
  }

  protected static FeatureNode[][] initLibLinear(
      final BagOfBigrams[] bob,
      final Dictionary dict) {

    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<>(bop.bob.size());
      for (LongIntCursor word : bop.bob) {
        if (word.value > 0) {
          features.add(new FeatureNode(dict.getWordIndex(word.key), word.value));
        }
      }
      FeatureNode[] featuresArray = features.toArray(new FeatureNode[]{});
      Arrays.parallelSort(featuresArray, new Comparator<FeatureNode>() {
        public int compare(FeatureNode o1, FeatureNode o2) {
          return Integer.compare(o1.index, o2.index);
        }
      });
      featuresTrain[j] = featuresArray;
    }
    return featuresTrain;
  }

  protected static double[] getLabels(final BagOfBigrams[] bagOfPatternsTestSamples) {
    double[] labels = new double[bagOfPatternsTestSamples.length];
    for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
      labels[i] = Double.valueOf(bagOfPatternsTestSamples[i].label);
    }
    return labels;
  }

  public WEASELModel getModel() {
    return model;
  }

  public void setModel(WEASELModel model) {
    this.model = model;
  }
}
