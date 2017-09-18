// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.WEASEL;
import sfa.transformation.WEASEL.BagOfBigrams;
import sfa.transformation.WEASEL.Dictionary;

import com.carrotsearch.hppc.cursors.IntIntCursor;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

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
  public static double p = 0.1;
  public static int iterations = 5000;
  public static double c = 1;

  // the trained weasel
  WEASELModel model;

  public WEASELClassifier() {
    super();
  }

  public static class WEASELModel extends Model {
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
      super("Weasel", testing, testSize, training, trainSize, normed, -1);
      this.features = features;
      this.weasel = model;
      this.linearModel = linearModel;
    }

    public int features;
    public WEASEL weasel;
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
      outputResult((int) score.training, startTime, trainSamples.length);
    }

    // determine label based on the majority of predictions
    int correctTesting = score(testSamples).correct.get();

    if (DEBUG) {
      System.out.println("Weasel Testing:\t");
      outputResult(correctTesting, startTime, testSamples.length);
      System.out.println("");
    }

    return new Score(
            "Weasel",
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
    String[] labels = predict(testSamples);
    return evalLabels(testSamples, labels);
  }

  private Predictions evalLabels(TimeSeries[] testSamples, String[] labels) {
    int correct = 0;
    for (int ind = 0; ind < testSamples.length; ind++) {
      correct+=compareLabels(labels[ind],(testSamples[ind].getLabel()))?1:0;
    }
    return new Predictions(labels, correct);
  }

  public String[] predict(TimeSeries[] samples) {
    final int[][][] wordsTest = model.weasel.createWords(samples);
    BagOfBigrams[] bagTest = model.weasel.createBagOfPatterns(wordsTest, samples, model.features);

    // chi square changes key mappings => remap
    model.weasel.dict.remap(bagTest);

    FeatureNode[][] features = initLibLinear(bagTest, model.linearModel.getNrFeature());

    String[] labels = new String[samples.length];

    for (int ind = 0; ind < features.length; ind++) {
      double label = Linear.predict(model.linearModel, features[ind]);
      labels[ind] = String.valueOf(label);
    }
    return labels;
  }

  protected WEASELModel fitWeasel(final TimeSeries[] samples) {
    try {
      int maxCorrect = -1;
      int bestF = -1;
      boolean bestNorm = false;

      int min = 4;
      int max = getMax(samples, MAX_WINDOW_LENGTH);
      int[] windowLengths = new int[max - min + 1];
      for (int w = min, a = 0; w <= max; w++, a++) {
        windowLengths[a] = w;
      }

      optimize:
      for (final boolean mean : NORMALIZATION) {
        WEASEL model = new WEASEL(maxF, maxS, windowLengths, mean, false);
        int[][][] words = model.createWords(samples);

        for (int f = minF; f <= maxF; f += 2) {
          model.dict.reset();
          BagOfBigrams[] bop = model.createBagOfPatterns(words, samples, f);
          model.filterChiSquared(bop, chi);

          // train liblinear
          final Problem problem = initLibLinearProblem(bop, model.dict, bias);
          int correct = trainLibLinear(problem, solverType, c, iterations, p, folds, new Random(1));

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
      WEASEL model = new WEASEL(maxF, maxS, windowLengths, bestNorm, false);
      int[][][] words = model.createWords(samples);
      BagOfBigrams[] bob = model.createBagOfPatterns(words, samples, bestF);
      model.filterChiSquared(bob, chi);

      // train liblinear
      Problem problem = initLibLinearProblem(bob, model.dict, bias);
      Linear.resetRandom();
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));

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

  protected static Problem initLibLinearProblem(
          final BagOfBigrams[] bob,
          final Dictionary dict,
          final double bias) {
    Problem problem = new Problem();
    problem.bias = bias;
    problem.n = dict.size() + 1;
    problem.y = getLabels(bob);

    final FeatureNode[][] features = initLibLinear(bob, problem.n);

    problem.l = features.length;
    problem.x = features;
    return problem;
  }

  protected static FeatureNode[][] initLibLinear(final BagOfBigrams[] bob, int max_feature) {
    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<>(bop.bob.size());
      for (IntIntCursor word : bop.bob) {
        if (word.value > 0 && word.key <= max_feature) {
          features.add(new FeatureNode(word.key, (word.value)));
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


  @SuppressWarnings("static-access")
  protected static int trainLibLinear(
          final Problem prob, final SolverType solverType, double c,
          int iter, double p, int nr_fold, Random random) {
    final Parameter param = new Parameter(solverType, c, iter, p);

    int i;
    final int l = prob.l;
    final int[] perm = new int[l];

    if (nr_fold > l) {
      nr_fold = l;
    }
    final int[] fold_start = new int[nr_fold + 1];

    for (i = 0; i < l; i++) {
      perm[i] = i;
    }
    for (i = 0; i < l; i++) {
      int j = i + random.nextInt(l - i);
      swap(perm, i, j);
    }
    for (i = 0; i <= nr_fold; i++) {
      fold_start[i] = i * l / nr_fold;
    }

    final AtomicInteger correct = new AtomicInteger(0);

    final int fold = nr_fold;
    ParallelFor.withIndex(threads, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        ThreadLocal<Linear> myLinear = new ThreadLocal<>();
        myLinear.set(new Linear());
        myLinear.get().disableDebugOutput();
        myLinear.get().resetRandom(); // reset random component of liblinear for reproducibility

        for (int i = 0; i < fold; i++) {
          if (i % threads == id) {

            int begin = fold_start[i];
            int end = fold_start[i + 1];
            int j, k;
            Problem subprob = new Problem();

            subprob.bias = prob.bias;
            subprob.n = prob.n;
            subprob.l = l - (end - begin);
            subprob.x = new Feature[subprob.l][];
            subprob.y = new double[subprob.l];

            k = 0;
            for (j = 0; j < begin; j++) {
              subprob.x[k] = prob.x[perm[j]];
              subprob.y[k] = prob.y[perm[j]];
              ++k;
            }
            for (j = end; j < l; j++) {
              subprob.x[k] = prob.x[perm[j]];
              subprob.y[k] = prob.y[perm[j]];
              ++k;
            }

            de.bwaldvogel.liblinear.Model submodel = myLinear.get().train(subprob, param);
            for (j = begin; j < end; j++) {
              correct.addAndGet(prob.y[perm[j]] == myLinear.get().predict(submodel, prob.x[perm[j]]) ? 1 : 0);
            }
          }
        }
      }
    });
    return correct.get();
  }

  private static void swap(int[] array, int idxA, int idxB) {
    int temp = array[idxA];
    array[idxA] = array[idxB];
    array[idxB] = temp;
  }

  protected static double[] getLabels(final BagOfBigrams[] bagOfPatternsTestSamples) {
    double[] labels = new double[bagOfPatternsTestSamples.length];
    for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
      labels[i] = Double.valueOf(bagOfPatternsTestSamples[i].label);
    }
    return labels;
  }

}