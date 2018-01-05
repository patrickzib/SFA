// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import com.carrotsearch.hppc.cursors.IntIntCursor;
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
  public static int minF = 4;
  public static int maxS = 4;
  public static SFA.HistogramType[] histTypes
      = new SFA.HistogramType[]{SFA.HistogramType.EQUI_DEPTH, SFA.HistogramType.EQUI_FREQUENCY};

  public static double chi = 2;
  public static double bias = 1;
  public static SolverType solverType = SolverType.L2R_LR_DUAL;
  public static int iterations = 1000;
  public static double p = 0.1;
  public static double c = 1;

  public static boolean BIGRAMS = true;
  public static boolean lowerBounding = true;

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

      int min = 4;
      int max = getMax(samples, MAX_WINDOW_LENGTH);
      final int[] windowLengths = new int[max - min + 1];
      for (int w = min, a = 0; w <= max; w++, a++) {
        windowLengths[a] = w;
      }

      optimize:
      for (final SFA.HistogramType histType : histTypes) {
        for (final boolean mean : NORMALIZATION) {
          final MUSE model = new MUSE(maxF, maxS, histType, windowLengths, mean, lowerBounding);
          final int[][][] words = model.createWords(samples);
          for (int f = minF; f <= maxF; f += 2) {
            model.dict.reset();
            MUSE.BagOfBigrams[] bag = model.createBagOfPatterns(words, samples, dimensionality, f);
            model.filterChiSquared(bag, chi);

            // train liblinear
            final Problem problem = initLibLinearProblem(bag, model.dict, bias);
            int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);

            if (correct > maxCorrect) {
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


      // obtain the final matrix
      MUSE model = new MUSE(bestF, maxS, bestHistType, windowLengths, bestNorm, lowerBounding);
      int[][][] words = model.createWords(samples);
      MUSE.BagOfBigrams[] bob = model.createBagOfPatterns(words, samples, dimensionality, bestF);
      model.filterChiSquared(bob, chi);

      // train liblinear
      Problem problem = initLibLinearProblem(bob, model.dict, bias);
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));

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

  public Double[] predict(final MultiVariateTimeSeries[] samples) {
    // iterate each sample to classify
    int dimensionality = samples[0].getDimensions();
    final int[][][] wordsTest = model.muse.createWords(samples);
    MUSE.BagOfBigrams[] bagTest = model.muse.createBagOfPatterns(wordsTest, samples, dimensionality, model.features);

    // chi square changes key mappings => remap
    model.muse.dict.remap(bagTest);

    FeatureNode[][] features = initLibLinear(bagTest, model.linearModel.getNrFeature());

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
    problem.n = dict.size() + 2 + 1;
    problem.y = getLabels(bob);

    final FeatureNode[][] features = initLibLinear(bob, problem.n);
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

  protected static FeatureNode[][] initLibLinear(final MUSE.BagOfBigrams[] bob, int max_feature) {
    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      MUSE.BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<FeatureNode>(bop.bob.size());
      for (IntIntCursor word : bop.bob) {
        if (word.value > 0 && word.key <= max_feature) {
          features.add(new FeatureNode(word.key, word.value));
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