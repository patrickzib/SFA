// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import com.carrotsearch.hppc.cursors.LongIntCursor;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import sfa.timeseries.TimeSeries;
import sfa.transformation.WEASELCharacter;
import sfa.transformation.WEASELCharacter.BagOfBigrams;
import sfa.transformation.WEASELCharacter.Dictionary;
import subwordTransformer.no.NoParameter;
import subwordTransformer.no.NoTransformer;

/**
 * The WEASEL (Word ExtrAction for time SEries cLassification) classifier as
 * published in
 * <p>
 * Schäfer, P., Leser, U.: Fast and Accurate Time Series Classification with
 * WEASEL." CIKM 2017
 */
public class WEASELCharacterClassifier extends Classifier {

  // default training parameters
  public static int maxF = 6;
  public static int minF = 4;
  public static int maxS = 4;

  public static SolverType solverType = SolverType.L2R_LR_DUAL;

  public static double chi = 2;
  public static double bias = 1;
  public static double p = 0.1;
  public static int iterations = 5000;
  public static double c = 1;

  public static boolean lowerBounding = false;

  public static int MIN_WINDOW_LENGTH = 2;
  public static int MAX_WINDOW_LENGTH = 350;

  public static NoTransformer transformer = new NoTransformer(maxS);
  public static List<NoParameter> transformerParameterList = NoParameter.getParameterList();

  // the trained weasel
  WEASELCharacterModel model;

  public WEASELCharacterClassifier() {
    super();
    Linear.resetRandom();
  }

  public static class WEASELCharacterModel extends Model {

    public WEASELCharacterModel() {
    }

    public WEASELCharacterModel(boolean normed, int features, WEASELCharacter model, de.bwaldvogel.liblinear.Model linearModel, int testing, int testSize, int training, int trainSize) {
      super("WEASELCharacter", testing, testSize, training, trainSize, normed, -1);
      this.features = features;
      this.weasel = model;
      this.linearModel = linearModel;
    }

    // the best number of Fourier values to be used
    public int features;

    // the trained WEASEL transformation
    public WEASELCharacter weasel;

    // the trained liblinear classifier
    public de.bwaldvogel.liblinear.Model linearModel;
  }

  @Override
  public Score eval(final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
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
      System.out.println("WEASELCharacter Testing:\t");
      outputResult(correctTesting, startTime, testSamples.length);
      System.out.println("");
    }

    return new Score("WEASELCharacter", correctTesting, testSamples.length, score.training, trainSamples.length, score.windowLength);
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

  @Override
  public Double[] predict(TimeSeries[] samples) {
    BagOfBigrams[] bagTest = null;

    for (int w = 0; w < model.weasel.windowLengths.length; w++) {
      final short[][][] words = model.weasel.createWords(samples, w);
      final int[][][] intSubwords = model.weasel.transformSubwordsOneWindow(words, w);

      BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(intSubwords, samples, w, model.features);
      model.weasel.dict.filterChiSquared(bopForWindow);
      bagTest = mergeBobs(bagTest, bopForWindow);
    }

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

    BagOfBigrams[] bagTest = null;
    for (int w = 0; w < model.weasel.windowLengths.length; w++) {
      final short[][][] words = model.weasel.createWords(samples, w);
      final int[][][] intSubwords = model.weasel.transformSubwordsOneWindow(words, w);

      BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(intSubwords, samples, w, model.features);
      model.weasel.dict.filterChiSquared(bopForWindow);
      bagTest = mergeBobs(bagTest, bopForWindow);
    }

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
    int min = norm && MIN_WINDOW_LENGTH <= 2 ? Math.max(3, MIN_WINDOW_LENGTH) : MIN_WINDOW_LENGTH;
    int max = getMax(samples, MAX_WINDOW_LENGTH);

    int[] wLengths = new int[max - min + 1];
    int a = 0;
    for (int w = min; w <= max; w += 1, a++) {
      wLengths[a] = w;
    }
    return Arrays.copyOfRange(wLengths, 0, a);
  }

  protected WEASELCharacterModel fitWeasel(final TimeSeries[] samples) {
    try {
      int maxCorrect = -1;
      int bestF = -1;
      boolean bestNorm = false;
      subwordTransformer.Parameter bestParam = null;

      optimize: for (final boolean mean : NORMALIZATION) {
        int[] windowLengths = getWindowLengths(samples, mean);
        WEASELCharacter model = new WEASELCharacter(maxF, maxS, windowLengths, mean, lowerBounding, transformer);
        short[][][][] words = model.createWords(samples); // TODO optional: also build words for each windowSize
        model.setTransformerTrainingWords(words);

        for (subwordTransformer.Parameter param : transformerParameterList) {
          model.fitSubwords(param);

          for (int f = minF; f <= maxF; f += 2) {
            model.dict.reset();

            BagOfBigrams[] bop = null;
            for (int w = 0; w < words.length; w++) {
              final int[][][] intSubwords = model.transformSubwordsOneWindow(words[w], w);

              BagOfBigrams[] bobForOneWindow = fitOneWindow(samples, windowLengths, mean, model, intSubwords, f, w);
              bop = mergeBobs(bop, bobForOneWindow);
            }

            // train liblinear
            final Problem problem = initLibLinearProblem(bop, model.dict, bias);
            int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);

            if (correct > maxCorrect) {
              // System.out.println(correct + "\t" + f);
              maxCorrect = correct;
              bestF = f;
              bestNorm = mean;
              bestParam = param;
            }
            if (correct == samples.length) {
              break optimize;
            }
          }
        }
      }

      // obtain the final matrix
      int[] windowLengths = getWindowLengths(samples, bestNorm);
      WEASELCharacter model = new WEASELCharacter(maxF, maxS, windowLengths, bestNorm, lowerBounding, transformer);
      short[][][][] words = model.createWords(samples);
      model.setTransformerTrainingWords(words);
      model.fitSubwords(bestParam);

      BagOfBigrams[] bob = null;
      for (int w = 0; w < model.windowLengths.length; w++) {
        final int[][][] intSubwords = model.transformSubwordsOneWindow(words[w], w);

        BagOfBigrams[] bobForOneWindow = fitOneWindow(samples, windowLengths, bestNorm, model, intSubwords, bestF, w);
        bob = mergeBobs(bob, bobForOneWindow);
      }

      // train liblinear
      Problem problem = initLibLinearProblem(bob, model.dict, bias);
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));

      return new WEASELCharacterModel(bestNorm, bestF, model, linearModel, 0, // testing
          1, maxCorrect, // training
          samples.length);

    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  private BagOfBigrams[] fitOneWindow(TimeSeries[] samples, int[] windowLengths, boolean mean, WEASELCharacter model, int[][][] word, int f, int w) {
    WEASELCharacter modelForWindow = new WEASELCharacter(maxF, maxS, windowLengths, mean, lowerBounding, transformer.getOutputAlphabetSize());
    BagOfBigrams[] bopForWindow = modelForWindow.createBagOfPatterns(word, samples, w, f);
    modelForWindow.filterChiSquared(bopForWindow, chi);

    // now, merge dicts
    model.dict.dictChi.putAll(modelForWindow.dict.dictChi);

    return bopForWindow;
  }

  private BagOfBigrams[] mergeBobs(BagOfBigrams[] bop, BagOfBigrams[] bopForWindow) {
    if (bop == null) {
      bop = bopForWindow;
    } else {
      for (int i = 0; i < bop.length; i++) {
        bop[i].bob.putAll(bopForWindow[i].bob);
      }
    }
    return bop;
  }

  protected static Problem initLibLinearProblem(final BagOfBigrams[] bob, final Dictionary dict, final double bias) {
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

  protected static FeatureNode[][] initLibLinear(final BagOfBigrams[] bob, final Dictionary dict) {

    dict.remap(bob);

    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<>(bop.bob.size());
      for (LongIntCursor word : bop.bob) {
        if (word.value > 0) {
          features.add(new FeatureNode((int) word.key, (word.value)));
        }
      }
      FeatureNode[] featuresArray = features.toArray(new FeatureNode[] {});
      Arrays.parallelSort(featuresArray, new Comparator<FeatureNode>() {
        @Override
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

  public WEASELCharacterModel getModel() {
    return model;
  }

  public void setModel(WEASELCharacterModel model) {
    this.model = model;
  }
}