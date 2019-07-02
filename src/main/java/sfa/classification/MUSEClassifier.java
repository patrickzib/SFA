// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import com.carrotsearch.hppc.cursors.IntIntCursor;

import com.carrotsearch.hppc.cursors.ObjectIntCursor;
import de.bwaldvogel.liblinear.*;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.transformation.MUSE;
import sfa.transformation.SFA;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

/**
 * The WEASEL+MUSE classifier as published in
 * <p>
 * Schäfer, P., Leser, U.: Multivariate Time Series Classification
 * with WEASEL+MUSE. arXiv 2017
 * http://arxiv.org/abs/1711.11343
 */
public class MUSEClassifier extends Classifier {

  public static int maxF = 6;
  public static int minF = 2;
  public static int maxS = 4;
  public static SFA.HistogramType[] histTypes
      = new SFA.HistogramType[]{SFA.HistogramType.EQUI_DEPTH, SFA.HistogramType.EQUI_FREQUENCY};

  public static double chi = 2;
  public static double bias = 1;
  public static SolverType solverType = SolverType.L2R_LR;
  public static int iterations = 5000;
  public static double p = 0.1;
  public static double c = 1;

  public static boolean BIGRAMS = true;
  public static boolean lowerBounding = false;

  public static int MIN_WINDOW_LENGTH = 2;
  public static int MAX_WINDOW_LENGTH = 450;

  // the trained muse model
  MUSEModel model;

  public MUSEClassifier() {
    super();
    TimeSeries.APPLY_Z_NORM = false; // FIXME static variable breaks some test cases!
  }

  public static class MUSEModel extends Model {

    public MUSEModel() {
    }

    public MUSEModel(
        boolean normed,
        int features,
        SFA.HistogramType histType,
        MUSE model,
        de.bwaldvogel.liblinear.Model linearModel,
        int testing,
        int testSize,
        int training,
        int trainSize
    ) {
      super("WEASEL+MUSE", testing, testSize, training, trainSize, normed, -1);
      this.features = features;
      this.muse = model;
      this.linearModel = linearModel;
      this.histType = histType;
    }

    // the best number of Fourier values to be used
    public int features;

    // the trained MUSE transformation
    public MUSE muse;

    // the trained liblinear classifier
    public de.bwaldvogel.liblinear.Model linearModel;

    public SFA.HistogramType histType;
  }

  @Override
  public Score eval(
      final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
    throw new RuntimeException("Please use: eval(" +
        "final MultiVariateTimeSeries[] trainSamples, final MultiVariateTimeSeries[] testSamples)");
  }

  @Override
  public Double[] predict(TimeSeries[] samples) {
    throw new RuntimeException("Please use: predict(final MultiVariateTimeSeries[] samples)");
  }

  @Override
  public Score fit(final TimeSeries[] trainSamples) {
    throw new RuntimeException("Please use: fit(final MultiVariateTimeSeries[] trainSamples)");
  }

  @Override
  public Predictions score(final TimeSeries[] testSamples) {
    throw new RuntimeException("Please use: score(final MultiVariateTimeSeries[] testSamples)");
  }

  public Score eval(
      final MultiVariateTimeSeries[] trainSamples, final MultiVariateTimeSeries[] testSamples) {
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
      System.out.println("WEASEL+MUSE Testing:\t");
      outputResult(correctTesting, startTime, testSamples.length);
      System.out.println("");
    }

    return new Score(
        "WEASEL+MUSE",
        correctTesting, testSamples.length,
        score.training, trainSamples.length,
        score.windowLength);
  }

  public Score fit(final MultiVariateTimeSeries[] trainSamples) {
    // train the shotgun models for different window lengths
    this.model = fitMuse(trainSamples);

    // return score
    return model.score;
  }


  public Predictions score(final MultiVariateTimeSeries[] testSamples) {
    Double[] labels = predict(testSamples);
    return evalLabels(testSamples, labels);
  }


  public int getMax(MultiVariateTimeSeries[] samples, int MAX_WINDOW_SIZE) {
    int max = samples[0].timeSeries[0].getLength();
    for (MultiVariateTimeSeries mts : samples) {
      for (TimeSeries ts : mts.timeSeries) {
        max = Math.max(ts.getLength(), max);
      }
    }
    return Math.min(max, MAX_WINDOW_SIZE);
  }

  public MUSEModel fitMuse(final MultiVariateTimeSeries[] samples) {
    int dimensionality = samples[0].getDimensions();
    try {
      int maxCorrect = -1;
      int bestF = -1;
      boolean bestNorm = false;
      SFA.HistogramType bestHistType = null;

      optimize:
      for (final SFA.HistogramType histType : histTypes) {
        for (final boolean mean : NORMALIZATION) {
          int[] windowLengths = getWindowLengths(samples, mean);

          for (int f = minF; f <= maxF; f += 2) {
            final MUSE model = new MUSE(f, maxS, histType, windowLengths, mean, lowerBounding);
            MUSE.BagOfBigrams[] bag = null;

            for (int w = 0; w < model.windowLengths.length; w++) {
              int[][] words = model.createWords(samples, w);
              MUSE.BagOfBigrams[] bobForOneWindow = fitOneWindow(
                  samples,
                  windowLengths, mean, histType,
                  model,
                  words, f, dimensionality, w);
              bag = mergeBobs(bag, bobForOneWindow);
            }

            // train liblinear
            final Problem problem = initLibLinearProblem(bag, model.dict, bias);
            int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);

            if (correct > maxCorrect || correct == maxCorrect && f < bestF) {
              maxCorrect = correct;
              bestF = f;
              bestNorm = mean;
              bestHistType = histType;

              if (DEBUG) {
                System.out.println("New best model" + maxCorrect + " " + bestF + " " + bestNorm + " " + bestHistType);
              }
            }
            if (correct == samples.length) {
              break optimize;
            }
          }
        }
      }

      final int[] windowLengths = getWindowLengths(samples, bestNorm);

      // obtain the final matrix
      MUSE model = new MUSE(bestF, maxS, bestHistType, windowLengths, bestNorm, lowerBounding);
      MUSE.BagOfBigrams[] bob = null;

      for (int w = 0; w < model.windowLengths.length; w++) {
        int[][] words = model.createWords(samples, w);

        MUSE.BagOfBigrams[] bobForOneWindow = fitOneWindow(
            samples,
            windowLengths, bestNorm, bestHistType,
            model,
            words,
            bestF, dimensionality, w);
        bob = mergeBobs(bob, bobForOneWindow);
      }

      // train liblinear
      Problem problem = initLibLinearProblem(bob, model.dict, bias);
      Parameter par = new Parameter(solverType, c, iterations, p);
      //par.setThreadCount(Math.min(Runtime.getRuntime().availableProcessors(),10));
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, par);

      return new MUSEModel(
          bestNorm,
          bestF,
          bestHistType,
          model,
          linearModel,
          0,
          1,
          maxCorrect,
          samples.length);

    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  private MUSE.BagOfBigrams[] fitOneWindow(
      MultiVariateTimeSeries[] samples,
      int[] windowLengths, boolean mean,
      SFA.HistogramType histType,
      MUSE model,
      int[][] word, int f, int dimensionality, int w) {
    MUSE modelForWindow = new MUSE(f, maxS, histType, windowLengths, mean, lowerBounding);

    MUSE.BagOfBigrams[] bopForWindow = modelForWindow.createBagOfPatterns(word, samples, w, dimensionality, f);
    modelForWindow.filterChiSquared(bopForWindow, chi);

    // now, merge dicts
    model.dict.dictChi.putAll(modelForWindow.dict.dictChi);

    return bopForWindow;
  }

  private MUSE.BagOfBigrams[] mergeBobs(
      MUSE.BagOfBigrams[] bop,
      MUSE.BagOfBigrams[] bopForWindow) {
    if (bop == null) {
      bop = bopForWindow;
    } else {
      for (int i = 0; i < bop.length; i++) {
        bop[i].bob.putAll(bopForWindow[i].bob);
      }
    }
    return bop;
  }

  public int[] getWindowLengths(final MultiVariateTimeSeries[] samples, boolean norm) {
    int min = norm && MIN_WINDOW_LENGTH<=2? Math.max(3,MIN_WINDOW_LENGTH) : MIN_WINDOW_LENGTH;
    int max = getMax(samples, MAX_WINDOW_LENGTH);
    final int[] windowLengths = new int[max - min + 1];
    for (int w = min, a = 0; w <= max; w++, a++) {
      windowLengths[a] = w;
    }
    return windowLengths;
  }

  public Double[] predict(final MultiVariateTimeSeries[] samples) {
    // iterate each sample to classify
    int dimensionality = samples[0].getDimensions();

    MUSE.BagOfBigrams[] bagTest = null;
    for (int w = 0; w < model.muse.windowLengths.length; w++) {
      int[][] wordsTest = model.muse.createWords(samples, w);
      MUSE.BagOfBigrams[] bopForWindow = model.muse.createBagOfPatterns(wordsTest, samples, w, dimensionality, model.features);
      model.muse.dict.filterChiSquared(bopForWindow);
      bagTest = mergeBobs(bagTest, bopForWindow);
    }

    FeatureNode[][] features = initLibLinear(bagTest, model.muse.dict);

    Double[] labels = new Double[samples.length];
    for (int ind = 0; ind < features.length; ind++) {
      double label = Linear.predict(model.linearModel, features[ind]);
      labels[ind] = label;
    }

    return labels;
  }

  public static Problem initLibLinearProblem(
      final MUSE.BagOfBigrams[] bob, final MUSE.Dictionary dict, final double bias) {
    Linear.resetRandom();

    Problem problem = new Problem();
    problem.bias = bias;
    problem.y = getLabels(bob);

    final FeatureNode[][] features = initLibLinear(bob, dict);

    problem.n = dict.size() + 1;
    problem.l = features.length;
    problem.x = features;
    return problem;
  }

  public static double[] getLabels(final MUSE.BagOfBigrams[] bagOfPatternsTestSamples) {
    double[] labels = new double[bagOfPatternsTestSamples.length];
    for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
      labels[i] = bagOfPatternsTestSamples[i].label;
    }
    return labels;
  }

  protected static FeatureNode[][] initLibLinear(
      final MUSE.BagOfBigrams[] bob,
      final MUSE.Dictionary dict) {

    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      MUSE.BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<FeatureNode>(bop.bob.size());
      for (ObjectIntCursor<MUSE.MuseWord> word : bop.bob) {
        if (word.value > 0 ) {
          features.add(new FeatureNode(dict.getWordChi(word.key), word.value));
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

}
