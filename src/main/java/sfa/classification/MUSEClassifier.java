// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import com.carrotsearch.hppc.cursors.IntLongCursor;
import de.bwaldvogel.liblinear.*;
import sfa.SFAWordsTest;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.MUSE;
import sfa.transformation.SFA;
import sfa.transformation.WEASEL;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

/**
 * TODO
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
  public static int iterations = 5000;
  public static double p = 0.1;
  public static double c = 1;

  // the trained weasel
  MUSEModel model;

  public static boolean lowerBounding = false;

  public static final int MAX_WINDOW_SIZE = 450;

  public MUSEClassifier() {
    super();
    Linear.resetRandom();
  }

//  public MUSEClassifier(MultiVariateTimeSeries[] train, MultiVariateTimeSeries[] test, boolean useDeltas) throws IOException {
//    super();
//
//    if (useDeltas) {
//      this.mtsTrainSamples = getDerivatives(train);
//      this.mtsTestSamples = getDerivatives(test);
//    }
//    else {
//      this.mtsTestSamples = test;
//      this.mtsTrainSamples = train;
//    }
//    dimensionality = this.mtsTrainSamples[0].getDimensions();
//  }


  public static class MUSEModel extends Model {

    public MUSEModel() {
    }

    public MUSEModel(
        boolean normed,
        int features,
        MUSE model,
        de.bwaldvogel.liblinear.Model linearModel,
        int testing,
        int testSize,
        int training,
        int trainSize
    ) {
      super("WEASEL", testing, testSize, training, trainSize, normed, -1);
      this.features = features;
      this.muse = model;
      this.linearModel = linearModel;
    }

    // the best number of Fourier values to be used
    public int features;

    // the trained MUSE transformation
    public MUSE muse;

    // the trained liblinear classifier
    public de.bwaldvogel.liblinear.Model linearModel;
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
      outputResult((int) score.training, startTime, trainSamples.length);
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
      int max = getMax(samples, MAX_WINDOW_SIZE);
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

  public Double[] predict(final MultiVariateTimeSeries[] testSamples) {
    // iterate each sample to classify
    int dimensionality = testSamples[0].getDimensions();
    final int[][][] wordsTest = model.muse.createWords(testSamples);
    MUSE.BagOfBigrams[] bagTest = model.muse.createBagOfPatterns(wordsTest, testSamples, dimensionality, model.features);

    // chi square changes key mappings => remap
    model.muse.dict.remap(bagTest);

    FeatureNode[][] features = initLibLinear(bagTest);

    Double[] labels = new Double[testSamples.length];

    for (int ind = 0; ind < features.length; ind++) {
      double label = Linear.predict(model.linearModel, features[ind]);
      labels[ind] = label;
    }

    return labels;
  }

  public static Problem initLibLinearProblem(
      final MUSE.BagOfBigrams[] bob, final MUSE.Dictionary dict, final double bias) {
    Linear.resetRandom();
    final FeatureNode[][] features = initLibLinear(bob);

    Problem problem = new Problem();
    problem.bias = bias;
    problem.n = dict.size() + 2 + 1;
    problem.y = getLabels(bob);
    problem.l = features.length;
    problem.x = features;

    return problem;
  }

  public static double[] getLabels(final MUSE.BagOfBigrams[] bagOfPatternsTestSamples) {
    double[] labels = new double[bagOfPatternsTestSamples.length];
    for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
      labels[i] = Double.valueOf(bagOfPatternsTestSamples[i].label);
    }
    return labels;
  }

  // FIXME : maxFeature??? refactor with WEASELClassifier
  public static FeatureNode[][] initLibLinear(final MUSE.BagOfBigrams[] bob) {
    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      MUSE.BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<FeatureNode>(bop.bob.size());

      for (IntLongCursor word : bop.bob) {
        if (word.value > 0) {
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

  protected static MultiVariateTimeSeries[] getDerivatives(MultiVariateTimeSeries[] mtsSamples) {
    for (MultiVariateTimeSeries mts : mtsSamples) {
      TimeSeries[] deltas = new TimeSeries[2 * mts.timeSeries.length];
      TimeSeries[] samples = mts.timeSeries;
      for (int a = 0; a < samples.length; a++) {
        TimeSeries s = samples[a];
        double[] d = new double[s.getLength() - 1];
        for (int i = 1; i < s.getLength(); i++) {
          d[i - 1] = s.getData()[i] - s.getData()[i - 1];
        }
        deltas[2 * a] = samples[a];
        deltas[2 * a + 1] = new TimeSeries(d, mts.getLabel());
      }
      mts.timeSeries = deltas;
    }
    return mtsSamples;
  }

//  protected static void extend(TimeSeries[] samples) {
//    for (int a = 0; a < samples.length; a++) {
//      TimeSeries s = samples[a];
//      int ext = 10;
//      double[] d = new double[s.getSize() + 2 * ext];
//      for (int i = 0; i < ext; i++) {
//        d[i] = s.getData(0);
//      }
//      for (int i = 0; i < s.getSize(); i++) {
//        d[i + ext] = s.getData(i);
//      }
//      for (int i = 0; i < ext; i++) {
//        d[i + ext + s.getSize()] = s.getData(s.getSize() - 1);
//      }
//      samples[a] = new TimeSeries(d, d.length, s.getLabel());
//    }
//  }


  public static String[] datasets = new String[]{
      "LP1",
      "LP2",
      "LP3",
      "LP4",
      "LP5",
      "PenDigits",
      "ShapesRandom",
      "DigitShapeRandom",
      "CMUsubject16",
      "ECG",
      "JapaneseVowels",
      "KickvsPunch",
      "Libras",
      "UWave",
      "Wafer",
      "WalkvsRun",
      "CharacterTrajectories",
      "ArabicDigits",
      "AUSLAN",
      "NetFlow",
  };

  public static void main(String argv[]) throws IOException {
    try {
      // the relative path to the datasets
      ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

      File dir = new File(classLoader.getResource("datasets/multivariate/").getFile());

      for (String s : datasets) {
        File d = new File(dir.getAbsolutePath() + "/" + s);
        if (d.exists() && d.isDirectory()) {
          for (File train : d.listFiles()) {
            if (train.getName().toUpperCase().endsWith("TRAIN3")) {
              File test = new File(train.getAbsolutePath().replaceFirst("TRAIN3", "TEST3"));

              if (!test.exists()) {
                System.err.println("File " + test.getName() + " does not exist");
                test = null;
              }

              Classifier.DEBUG = false;

              MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDatset(train);
              MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDatset(test);

              boolean useDeltas = true;
              if (useDeltas) {
                trainSamples = getDerivatives(trainSamples);
                testSamples = getDerivatives(testSamples);
              }

              MUSEClassifier weasel = new MUSEClassifier();
              MUSEClassifier.lowerBounding = true;
              MUSEClassifier.folds = 10;
              MUSEClassifier.NORMALIZATION = new boolean[]{true, false};

              MUSEClassifier.Score weaselScore = weasel.eval(trainSamples, testSamples);
              System.out.println(s + ";" + weaselScore.toString());
            }
          }
        }
      }
    } finally {
      ParallelFor.shutdown();
    }
  }
}