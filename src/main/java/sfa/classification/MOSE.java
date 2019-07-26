package sfa.classification;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.DoubleDoubleCursor;
import com.carrotsearch.hppc.cursors.LongIntCursor;
import com.carrotsearch.hppc.cursors.ObjectCursor;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import libsvm.*;
import sfa.SFAWordsTest;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.WEASEL;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 */
public class MOSE extends WEASELClassifier {

  ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
  File sequence = new File(classLoader.getResource("datasets/sequences/TEK16.txt_TRAIN").getFile());

  public MOSE() {
    super();
  }

  public static class MOSEModel extends WEASELModel {
    public MOSEModel(){}

    public MOSEModel(
        boolean normed,
        int features,
        WEASEL model,
        de.bwaldvogel.liblinear.Model linearModel,
        WEASEL.BagOfBigrams[] bop,
        int testing,
        int testSize,
        int training,
        int trainSize
    ) {
      super(normed, features, model, linearModel, testing, testSize, training, trainSize);
      this.bop = bop;
      this.name = "MOSE";
    }

    public WEASEL.BagOfBigrams[] bop;
  }

  public void eval() {

    //maxF = 6;
    MIN_WINDOW_LENGTH = 100;
    MAX_WINDOW_LENGTH = 200;
    WEASEL.WORD_LIMIT = 10;
    solverType = SolverType.L2R_L2LOSS_SVC;
    c = 1;
    chi = 1;
    maxS = 8;
    maxF = 12;

    TimeSeries sample = TimeSeriesLoader.loadDataset(sequence)[0];
    fitMose(sample);
  }

  public void fitMose(
      final TimeSeries sample) {
    try {

      double[] data = sample.getData();
      int width = Math.min(500, data.length/10);
      System.out.println("Width: " + width);

      TimeSeries[] disjointTS = sample.getDisjointSequences(width, true);

      MIN_WINDOW_LENGTH = width / 4;
      MAX_WINDOW_LENGTH = (int)(width / 1.5);
      maxF = (int)Math.sqrt(MAX_WINDOW_LENGTH);

      System.out.println("Features: " + maxF);

      System.out.println("MIN_WINDOW_LENGTH:" + MIN_WINDOW_LENGTH);
      System.out.println("MAX_WINDOW_LENGTH:" + MAX_WINDOW_LENGTH);

      for (int i = 0; i < disjointTS.length; i++) {
        //disjointTS[i].setLabel(i >= disjointTS.length/2 ? -1.0 : 1.0);
        disjointTS[i].setLabel((double)i);
      }

      TimeSeries[] samples = disjointTS;

      System.out.println("Samples: " + samples.length);

//      int maxCorrect = Integer.MAX_VALUE;
//      int bestF = -1;
//      int bestS = -1;
//      boolean bestNorm = false;
//
//      optimize:
//      for (final boolean mean : NORMALIZATION) {
//        int[] windowLengths = getWindowLengths(samples, bestNorm);
//        //for (int s = minS; s <= maxS; s *= 2) {
//          WEASEL model = new WEASEL(maxF, maxS, windowLengths, mean, lowerBounding);
//          final int[][][] words = model.createWords(samples);
//
//          for (int f = minF; f <= maxF; f += 2) {
//            model.dict.reset();
//
//            final WEASEL.BagOfBigrams[] bop = new WEASEL.BagOfBigrams[samples.length];
//            final int ff = f;
//
//            ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
//              @Override
//              public void run(int id, AtomicInteger processed) {
//                for (int w = 0; w < model.windowLengths.length; w++) {
//                  if (w % BLOCKS == id) {
//                    WEASEL.BagOfBigrams[] bobForOneWindow = fitOneWindow(
//                        samples,
//                        model.windowLengths, mean,
//                        words[w], ff, w);
//                    mergeBobs(bop, bobForOneWindow);
//                  }
//                }
//              }
//            });
//
//            // train liblinear
//            final Problem problem = initLibLinearProblem(bop, model.dict, bias);
//            int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);
//
//            if (correct < maxCorrect) {
//              maxCorrect = correct;
//              bestF = f;
//              bestS = maxS;
//              bestNorm = mean;
//
//              System.out.println("Correct: " + correct);
//              System.out.println("bestNorm: " + mean);
//              System.out.println("BestF: " + bestF);
//              System.out.println("BestS: " + bestS);
//            }
//          }
//       // }
//      }

      // obtain the final matrix
      final boolean mean = true;
      int[] windowLengths = getWindowLengths(samples, mean);
      WEASEL model = new WEASEL(maxF, maxS, windowLengths, mean, lowerBounding);

      //int[][][] words = new int[model.windowLengths.length][][];
      final WEASEL.BagOfBigrams[] bop = new WEASEL.BagOfBigrams[samples.length];
      final int ff = maxF;
      ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
        @Override
        public void run(int id, AtomicInteger processed) {
          for (int w = 0; w < model.windowLengths.length; w++) {
            if (w % BLOCKS == id) {
              int[][] words = model.createWords(samples, w);
              WEASEL.BagOfBigrams[] bobForOneWindow = fitOneWindow(
                  samples,
                  model.windowLengths, mean,
                  words, (int)Math.max(4, Math.sqrt(model.windowLengths[w])), w);
              // TODO

              mergeBobs(bop, bobForOneWindow);
            }
          }
        }
      });

      ObjectHashSet<WEASEL.WeaselWord> usedWords = model.trainHighestCount(bop);

      // find used words
      HashSet<Integer> usedWindowLengths = new HashSet<>();
      for (ObjectCursor<WEASEL.WeaselWord> w : usedWords) {
        usedWindowLengths.add(w.value.w);
      }
      System.out.println("Used windows:"+usedWindowLengths);
      for (int w :  usedWindowLengths) {
        System.out.print(model.windowLengths[w] + " ");
      }
      System.out.println("");

      // train liblinear
      Problem problem = initLibLinearProblem(bop, model.dict, bias);
      //de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));
      //int correct = 0;
      //for (int j = 0; j < problem.x.length; j++) {
      //  correct += Linear.predict(linearModel, problem.x[j])==problem.y[j] ? 1 : 0;
      //}

      System.out.println("Train Dict Size: " + model.dict.size());

      // TODO
      //  - identify relevant windowSizes???
      //  - use actual weights???

      // obtain feature weights
      //double[] weights = linearModel.getFeatureWeights();



      // Generate the plot from the dictionary words

      TimeSeries[] slidingTs = sample.getSubsequences(width, true);
      double[] plot = new double[sample.getLength()];
      for (int w : usedWindowLengths) {
        int windowLength = model.windowLengths[w];

        // left split
        int[][] words = model.createWords(slidingTs, w);
        model.remapWords(words, slidingTs, w, maxF);

        for (int i = 0; i < words.length; i++) {
          for (int j = 0; j < words[i].length; j++) {
            WEASEL.WeaselWord word = new WEASEL.WeaselWord(w, words[i][j]);
            if (model.dict.dict.containsKey(word)) {
              int index = model.dict.getWordIndex(word);
              //double weight = weights[index];
              //int count = bob[i].bob.get(words[i][j]);

              for (int w2 = 0; w2 < windowLength; w2++) {
                plot[i + j + w2] = index; //, plot[i + j + w2];
              }

              // use the same color to the left
//              for (int w2 = i + j; w2 > 0; w2--) {
//                if (plot[w2] > 0)
//                  plot[w2] = index; //, plot[w2]);
//              }

              // use the same color to the right
//              for (int w2 = i + j + windowLength; w2 < plot.length; w2++) {
//                if (plot[w2] > 0)
//                  plot[w2] = index; //, plot[w2]);
//              }

            }
          }
        }
      }

      writeToDisc(plot);

      System.out.println("done");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public File getFilePath() {
    String dsName = sequence.getName();
    return new File("./heatmap_"+dsName);
  }


  public void writeToDisc(double[] plot) {
    File file = getFilePath();
    try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {

      for (int i = 0; i < plot.length; i++) {
        writer.write(String.format(Locale.ENGLISH, "%.1f", plot[i]));
        if (i < plot.length - 1) {
          writer.write(",");
        }
      }
    } catch(IOException e){
      e.printStackTrace();
    }
  }

  public static void main(String argv[]) {
    try {
      MOSE m = new MOSE();
      m.eval();
      m.exec.shutdown();
    } finally {
      ParallelFor.shutdown();
    }
  }
}