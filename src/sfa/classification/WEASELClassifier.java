package sfa.classification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.timeseries.TimeSeries;
import sfa.transformation.WEASELModel;
import sfa.transformation.WEASELModel.BagOfBigrams;
import sfa.transformation.WEASELModel.Dictionary;

import com.carrotsearch.hppc.cursors.IntIntCursor;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

/**
 * Liblinear: O(n) vs Libsvm: O(n^2) to O(n^3)
 * http://de.slideshare.net/previa/py-hug20130520-21510547
 *
 * http://cs.nyu.edu/~rostami/presentations/L1_vs_L2.pdf
 *
 * @author bzcschae
 *
 */
public class WEASELClassifier extends Classifier {

  public static int maxF = 6; // TODO used to be 6
  public static int minF = 4;
  public static int maxS = 4;

  public static SolverType solverType = SolverType.L2R_LR_DUAL;

  public static double chi = 2;
  public static double bias = 1;
  public static double p = 0.1;
  public static int iter = 5000;
  public static double c = 1;
  
  public WEASELClassifier(TimeSeries[] train, TimeSeries[] test) throws IOException {
    super(train, test);    
  }

  public static class WScore extends Score {
    public WScore(
        double testing,
        boolean normed,
        int features,
        WEASELModel model,
        Model linearModel
        ) {
      super("W", testing, testing, normed, -1);
      this.features = features;
      this.model = model;
      this.linearModel = linearModel;
    }

    public int features;
    public WEASELModel model;
    public Model linearModel;

    @Override
    public void clear() {
      super.clear();
      this.windowLength = -1;
      this.model = null;
      this.linearModel = null;
    }
  }

  public Score eval() throws IOException {
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    try {
      // generate test train/split for cross-validation
      generateIndices();

      long startTime = System.currentTimeMillis();
      this.correctTraining = new AtomicInteger(0);

      WScore bestScore = fit(this.trainSamples, exec);

      // training score
      if (DEBUG) {
        System.out.println("W Training:\t" + bestScore.windowLength + "," + bestScore.features + "," + bestScore.normed + "," + ", chi: " + chi);
        outputResult((int)bestScore.training, startTime, this.trainSamples.length);
      }

      // determine label based on the majority of predictions
      int correctTesting = predict(exec, bestScore, this.testSamples);

      if (DEBUG) {
        System.out.println("W Testing:\t");
        outputResult(correctTesting, startTime, this.testSamples.length);
        System.out.println("");
      }

      return new Score(
          "W",
          1-formatError(correctTesting, this.testSamples.length),
          1-formatError((int)bestScore.training, this.trainSamples.length),
          bestScore.normed,
          bestScore.windowLength
          );
    }
    finally {
      exec.shutdown();
    }

  }

  public WScore fit(
      final TimeSeries[] samples,
      final ExecutorService exec) {
    try {
      int maxCorrect = -1;
      int bestF = -1;
      boolean bestNorm = false;

      int min = 4;
      int max = getMax(samples, MAX_WINDOW_LENGTH);
      int [] windowLengths = new int[max - min + 1];
      for (int w = min, a=0; w <= max; w++, a++) {
        windowLengths[a] = w;
      }

      optimize:
        for (final boolean mean : NORMALIZATION) {
          WEASELModel model = new WEASELModel(maxF, maxS, windowLengths, mean, false);
          int[][][] words = model.createWords(samples);

          for (int f = minF; f <= maxF; f+=2) {
            model.dict.reset();
            BagOfBigrams[] bop = model.createBagOfPatterns(words, samples, f);
            model.filterChiSquared(bop, chi);

            // train liblinear
            final Problem problem = initLibLinearProblem(bop, model.dict, bias);
            int correct = trainLibLinear(problem, solverType, c, iter, p, folds, new Random(1));

            if (correct >  maxCorrect) {
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
      WEASELModel model = new WEASELModel(maxF, maxS, windowLengths, bestNorm, false);
      int[][][] words = model.createWords(samples);
      BagOfBigrams[] bob = model.createBagOfPatterns(words, samples, bestF);
      model.filterChiSquared(bob, chi);

      // train liblinear
      Problem problem = initLibLinearProblem(bob, model.dict, bias);
      Model linearModel = Linear.train(problem, new Parameter(solverType, c, iter, p));

      return new WScore(
          maxCorrect,
          bestNorm,
          bestF,
          model,
          linearModel);

    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  public int predict(
      final ExecutorService exec,
      final WScore score,
      final TimeSeries[] testSamples) {
    // iterate each sample to classify
    final int[][][] wordsTest = score.model.createWords(testSamples);
    BagOfBigrams[] bagTest = score.model.createBagOfPatterns(wordsTest, testSamples, score.features);

    // chi square changes key mappings => remap
    score.model.dict.remap(bagTest);

    FeatureNode[][] features = initLibLinear(bagTest);

    int correct = 0;
    for (int ind = 0; ind < features.length; ind++) {
      double label = Linear.predict(score.linearModel, features[ind]);
      if (label==Double.valueOf(bagTest[ind].label)) {
        correct++;
      }
    }
    return correct;
  }

  public static Problem initLibLinearProblem(
      final BagOfBigrams[] bob,
      final Dictionary dict,
      final double bias) {
    Linear.resetRandom();

    final FeatureNode[][] features = initLibLinear(bob);
    Problem problem = new Problem();
    problem.bias = bias;
    problem.n = dict.size()+1;
    problem.y = getLabels(bob);
    problem.l = features.length;
    problem.x = features;
    return problem;
  }

  public static FeatureNode[][] initLibLinear(final BagOfBigrams[] bob) {
    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<FeatureNode>(bop.bob.size());
      for (IntIntCursor word : bop.bob) {
        if (word.value > 0) {
          features.add(new FeatureNode(word.key, ((double)word.value)));
        }
      }
      FeatureNode[] featuresArray = features.toArray(new FeatureNode[]{});
      Arrays.parallelSort(featuresArray, new Comparator<FeatureNode> (){
        public int compare(FeatureNode o1, FeatureNode o2) {
          return Integer.compare(o1.index, o2.index);
        }
      });
      featuresTrain[j] = featuresArray;
    }
    return featuresTrain;
  }
  

  @SuppressWarnings("static-access")
  public static int trainLibLinear(
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
        ThreadLocal<Linear> myLinear = new ThreadLocal<Linear>();
        myLinear.set(new Linear());
        myLinear.get().disableDebugOutput();

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

            Model submodel = myLinear.get().train(subprob, param);
            for (j = begin; j < end; j++) {
              correct.addAndGet(prob.y[perm[j]] == myLinear.get().predict(submodel, prob.x[perm[j]]) ? 1 : 0);
            }
          }
        }
      }
    });
    return correct.get();
  }
  
  static void swap(int[] array, int idxA, int idxB) {
    int temp = array[idxA];
    array[idxA] = array[idxB];
    array[idxB] = temp;
  }
  
  public static double[] getLabels(final BagOfBigrams[] bagOfPatternsTestSamples) {
    double[] labels = new double[bagOfPatternsTestSamples.length];
    for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
      labels[i] = Double.valueOf(bagOfPatternsTestSamples[i].label);
    }
    return labels;
  }
  
}