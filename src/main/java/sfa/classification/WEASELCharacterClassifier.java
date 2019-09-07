// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

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
import subwordTransformer.SubwordTransformer;
import subwordTransformer.apriori.AprioriParameter;
import subwordTransformer.bpe.BPEParameter;
import subwordTransformer.bpe.SupervisedBPETransformer;
import subwordTransformer.cng.CNGParameter;

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

  // public static SubwordTransformer<?> transformer = new
  // SupervisedAprioriTransformer(maxS);
  // public static List<subwordTransformer.Parameter> transformerParameterList =
  // new ArrayList<>(AprioriParameter.getParameterList(2, 4, 0.5, 1, 1));

  public static SubwordTransformer<?> transformer = new SupervisedBPETransformer(maxS, true);
  public static List<subwordTransformer.Parameter> transformerParameterList = new ArrayList<>(BPEParameter.getParameterList(0.8, 1, 1));

  // public static SubwordTransformer<?> transformer = new
  // SupervisedCNGTransformer(maxS, true);
  // public static List<subwordTransformer.Parameter> transformerParameterList =
  // new ArrayList<>(CNGParameter.getParameterList(2, 4, 4, 6, 0.5, 1, 1));

  // the trained weasel
  WEASELCharacterModel model;

  public WEASELCharacterClassifier() {
    super();
    Linear.resetRandom();
    Linear.disableDebugOutput();
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
    final AtomicLong timeCreateWords = new AtomicLong(0);
    final AtomicLong timeTransform = new AtomicLong(0);
    final AtomicLong timeBOP = new AtomicLong(0);
    long time = System.currentTimeMillis();

    // iterate each sample to classify
    final BagOfBigrams[] bagTest = new BagOfBigrams[samples.length];
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < model.weasel.windowLengths.length; w++) {
          if (w % BLOCKS == id) {
            long subTime = System.currentTimeMillis();
            final short[][][] words = model.weasel.createWords(samples, w);
            timeCreateWords.addAndGet(System.currentTimeMillis() - subTime);

            subTime = System.currentTimeMillis();
            final int[][][] intSubwords = model.weasel.transformSubwordsOneWindow(words, w);
            timeTransform.addAndGet(System.currentTimeMillis() - subTime);

            subTime = System.currentTimeMillis();
            BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(intSubwords, samples, w, model.features);
            model.weasel.dict.filterChiSquared(bopForWindow);
            mergeBobs(bagTest, bopForWindow);
            timeBOP.addAndGet(System.currentTimeMillis() - subTime);
          }
        }
      }
    });

    time = System.currentTimeMillis() - time;
    System.out.println("test_createWords " + (long) (time * (1.0 * timeCreateWords.get() / (timeCreateWords.get() + timeTransform.get() + timeBOP.get()))));
    System.out.println("test_transformSubwords " + (long) (time * (1.0 * timeTransform.get() / (timeCreateWords.get() + timeTransform.get() + timeBOP.get()))));
    System.out.println("test_createBOP " + (long) (time * (1.0 * timeBOP.get() / (timeCreateWords.get() + timeTransform.get() + timeBOP.get()))));
    System.out.println("Memory: " + getUsedMemory() + " MB");

    time = System.currentTimeMillis();

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

    System.out.println("test_liblinear " + (System.currentTimeMillis() - time));
    System.out.println("Memory: " + getUsedMemory() + " MB");

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
            final short[][][] words = model.weasel.createWords(samples, w);
            final int[][][] intSubwords = model.weasel.transformSubwordsOneWindow(words, w);

            BagOfBigrams[] bopForWindow = model.weasel.createBagOfPatterns(intSubwords, samples, w, model.features);
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

      long time;
      optimize: for (final boolean mean : NORMALIZATION) {
        int[] windowLengths = getWindowLengths(samples, mean);
        WEASELCharacter model = new WEASELCharacter(maxF, maxS, windowLengths, mean, lowerBounding, transformer);

        time = System.currentTimeMillis();
        short[][][][] words = model.createWords(samples);
        System.out.println("train_createWords " + (System.currentTimeMillis() - time));
        System.out.println("Memory: " + getUsedMemory() + " MB");

        time = System.currentTimeMillis();
        model.setTransformerTrainingWords(words, samples);
        System.out.println("train_setTrainingWords " + (System.currentTimeMillis() - time));
        System.out.println("Memory: " + getUsedMemory() + " MB");

        for (subwordTransformer.Parameter param : transformerParameterList) {
          if (param instanceof CNGParameter) {
            CNGParameter cp = (CNGParameter) param;
            if (!((cp.getMinN() == 2 && cp.getMaxN() == 4) || (cp.getMinN() == 4 && cp.getMaxN() == 6)))
              continue;
          } else if (param instanceof AprioriParameter) {
            AprioriParameter ap = (AprioriParameter) param;
            if (!(ap.getMinSize() == 2 || ap.getMinSize() == 4))
              continue;
          }

          time = System.currentTimeMillis();
          model.fitSubwords(param);
          System.out.println("train_fitSubwords " + (System.currentTimeMillis() - time));
          System.out.println("Memory: " + getUsedMemory() + " MB");

          for (int f = minF; f <= maxF; f += 2) {
            if (param instanceof CNGParameter) {
              CNGParameter cp = (CNGParameter) param;
              if (cp.getMaxN() != f)
                continue;
            } else if (param instanceof AprioriParameter) {
              AprioriParameter ap = (AprioriParameter) param;
              if (ap.getMinSize() != f - 2)
                continue;
            }
            model.dict.reset();

            final AtomicLong timeTransform = new AtomicLong(0);
            final AtomicLong timeBOP = new AtomicLong(0);
            time = System.currentTimeMillis();

            final BagOfBigrams[] bop = new BagOfBigrams[samples.length];
            final int ff = f;

            ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
              @Override
              public void run(int id, AtomicInteger processed) {
                for (int w = 0; w < model.windowLengths.length; w++) {
                  if (w % BLOCKS == id) {
                    long subTime = System.currentTimeMillis();
                    final int[][][] intSubwords = model.transformSubwordsOneWindow(words[w], w);
                    timeTransform.addAndGet(System.currentTimeMillis() - subTime);

                    subTime = System.currentTimeMillis();
                    BagOfBigrams[] bobForOneWindow = fitOneWindow(samples, model.windowLengths, mean, intSubwords, ff, w);
                    mergeBobs(bop, bobForOneWindow);
                    timeBOP.addAndGet(System.currentTimeMillis() - subTime);
                  }
                }
              }
            });

            time = System.currentTimeMillis() - time;
            System.out.println("train_transformSubwords " + (long) (time * (1.0 * timeTransform.get() / (timeTransform.get() + timeBOP.get()))));
            System.out.println("train_createBOP " + (long) (time * (1.0 * timeBOP.get() / (timeTransform.get() + timeBOP.get()))));
            System.out.println("Memory: " + getUsedMemory() + " MB");

            time = System.currentTimeMillis();
            // train liblinear
            final Problem problem = initLibLinearProblem(bop, model.dict, bias);
            int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);

            System.out.println("train_liblinear " + (System.currentTimeMillis() - time));
            System.out.println("Memory: " + getUsedMemory() + " MB");

            System.out.println(correct + " correct with norm=" + mean + ", " + param + ", f=" + f);

            if (correct > maxCorrect) { // TODO > or >= ?
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

      System.out.println("Best parameter: norm=" + bestNorm + ", " + bestParam + ", f=" + bestF);

      // obtain the final matrix
      int[] windowLengths = getWindowLengths(samples, bestNorm);
      WEASELCharacter model = new WEASELCharacter(maxF, maxS, windowLengths, bestNorm, lowerBounding, transformer);

      time = System.currentTimeMillis();
      short[][][][] words = model.createWords(samples);
      System.out.println("train_createWords " + (System.currentTimeMillis() - time));
      System.out.println("Memory: " + getUsedMemory() + " MB");

      time = System.currentTimeMillis();
      model.setTransformerTrainingWords(words, samples);
      System.out.println("train_setTrainingWords " + (System.currentTimeMillis() - time));
      System.out.println("Memory: " + getUsedMemory() + " MB");

      time = System.currentTimeMillis();
      model.fitSubwords(bestParam);
      System.out.println("train_fitSubwords " + (System.currentTimeMillis() - time));
      System.out.println("Memory: " + getUsedMemory() + " MB");

      final AtomicLong timeTransform = new AtomicLong(0);
      final AtomicLong timeBOP = new AtomicLong(0);
      time = System.currentTimeMillis();

      final BagOfBigrams[] bop = new BagOfBigrams[samples.length];
      final boolean mean = bestNorm;
      final int ff = bestF;
      ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {

        @Override
        public void run(int id, AtomicInteger processed) {
          for (int w = 0; w < model.windowLengths.length; w++) {
            if (w % BLOCKS == id) {
              long subTime = System.currentTimeMillis();
              final int[][][] intSubwords = model.transformSubwordsOneWindow(words[w], w);
              timeTransform.addAndGet(System.currentTimeMillis() - subTime);

              subTime = System.currentTimeMillis();
              BagOfBigrams[] bobForOneWindow = fitOneWindow(samples, model.windowLengths, mean, intSubwords, ff, w);
              mergeBobs(bop, bobForOneWindow);
              timeBOP.addAndGet(System.currentTimeMillis() - subTime);
            }
          }
        }
      });

      time = System.currentTimeMillis() - time;
      System.out.println("train_transformSubwords " + (long) (time * (1.0 * timeTransform.get() / (timeTransform.get() + timeBOP.get()))));
      System.out.println("train_createBOP " + (long) (time * (1.0 * timeBOP.get() / (timeTransform.get() + timeBOP.get()))));
      System.out.println("Memory: " + getUsedMemory() + " MB");

      time = System.currentTimeMillis();
      // train liblinear
      Problem problem = initLibLinearProblem(bop, model.dict, bias);
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));

      System.out.println("train_liblinear " + (System.currentTimeMillis() - time));
      System.out.println("Memory: " + getUsedMemory() + " MB");

      int highestBit = Words.binlog(Integer.highestOneBit(MAX_WINDOW_LENGTH)) + 1;
      byte usedBits = (byte) Words.binlogRoundedUp(transformer.getOutputAlphabetSize());
      int shortsPerInt = 32 / usedBits;
      long subwordCount = 0;
      long wordCount = 0;
      long biGramSubwordCount = 0;
      long biGramCount = 0;
      for (LongIntCursor cursor : model.dict.dict) {
        if (cursor.value > 0) {
          long biGram = cursor.key;
          int prevWord = (int) (biGram >>> 32);
          int word = (int) ((biGram & 0xffffffffL) >>> highestBit);

          boolean wildcardFound = false;
          long shiftOffset = 1;
          for (int i = 0; i < shortsPerInt; i++) {
            boolean isWildcard = true;
            for (int j = 0; j < usedBits; j++) {
              if ((word & shiftOffset) == 0) {
                isWildcard = false;
              }
              shiftOffset <<= 1;
            }
            if (isWildcard) {
              wildcardFound = true;
              break;
            }
          }

          if (prevWord == 0) {
            wordCount++;
            if (wildcardFound) {
              subwordCount++;
            }
          } else {
            biGramCount++;

            if (wildcardFound) {
              biGramSubwordCount++;
            } else {
              shiftOffset = 1;
              for (int i = 0; i < shortsPerInt; i++) {
                boolean isWildcard = true;
                for (int j = 0; j < usedBits; j++) {
                  if ((prevWord & shiftOffset) == 0) {
                    isWildcard = false;
                  }
                  shiftOffset <<= 1;
                }
                if (isWildcard) {
                  biGramSubwordCount++;
                  break;
                }
              }
            }

          }
        }
      }
      System.out.println("dictSize " + model.dict.size() + "  subwords: unigrams " + subwordCount + "/" + wordCount + "  bigrams " + biGramSubwordCount + "/" + biGramCount);

      return new WEASELCharacterModel(bestNorm, bestF, model, linearModel, 0, // testing
          1, maxCorrect, // training
          samples.length);

    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  private BagOfBigrams[] fitOneWindow(TimeSeries[] samples, int[] windowLengths, boolean mean, int[][][] word, int f, int w) {
    WEASELCharacter modelForWindow = new WEASELCharacter(f, maxS, windowLengths, mean, lowerBounding, transformer.getOutputAlphabetSize());
    BagOfBigrams[] bopForWindow = modelForWindow.createBagOfPatterns(word, samples, w, f);
    modelForWindow.trainChiSquared(bopForWindow, chi);
    // modelForWindow.trainAnova(bopForWindow, chi);

    return bopForWindow;
  }

  private synchronized void mergeBobs(BagOfBigrams[] bop, BagOfBigrams[] bopForWindow) {
    for (int i = 0; i < bop.length; i++) {
      if (bop[i] == null) {
        bop[i] = bopForWindow[i];
      } else {
        bop[i].bob.putAll(bopForWindow[i].bob);
      }
    }
  }

  protected static Problem initLibLinearProblem(final BagOfBigrams[] bob, final Dictionary dict, final double bias) {
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

  protected static FeatureNode[][] initLibLinear(final BagOfBigrams[] bob, final Dictionary dict) {
    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<>(bop.bob.size());
      for (LongIntCursor word : bop.bob) {
        if (word.value > 0) {
          features.add(new FeatureNode(dict.getWordIndex(word.key), word.value));
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