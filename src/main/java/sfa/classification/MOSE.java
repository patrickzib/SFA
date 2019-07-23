package sfa.classification;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.DoubleDoubleCursor;
import com.carrotsearch.hppc.cursors.LongIntCursor;
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
  File sequence = new File(classLoader.getResource("datasets/sequences/brady45.txt_TRAIN").getFile());

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
    final boolean norm = true;
    maxF = 8;


    TimeSeries sample = TimeSeriesLoader.loadDataset(sequence)[0];
    fitMose(sample, norm);
  }

  public void fitMose(
      final TimeSeries sample, final boolean norm) {
    try {

      double[] data = sample.getData();
      int width = Math.min(500, data.length/10);

      TimeSeries[] disjointTS = sample.getDisjointSequences(width, norm);

      MIN_WINDOW_LENGTH = width / 8;
      MAX_WINDOW_LENGTH = width / 4;

      System.out.println("MIN_WINDOW_LENGTH:" + MIN_WINDOW_LENGTH);
      System.out.println("MAX_WINDOW_LENGTH:" + MAX_WINDOW_LENGTH);

      for (int i = 0; i < disjointTS.length; i++) {
        //disjointTS[i].setLabel(i >= disjointTS.length/2 ? -1.0 : 1.0);
        disjointTS[i].setLabel((double)i);
      }

      TimeSeries[] samples = disjointTS;

      // obtain the final matrix
      int[] windowLengths = getWindowLengths(samples, norm);
      WEASEL model = new WEASEL(maxF, maxS, windowLengths, norm, lowerBounding);

      //int[][][] words = new int[model.windowLengths.length][][];
      final WEASEL.BagOfBigrams[] bop = new WEASEL.BagOfBigrams[samples.length];
      ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
        @Override
        public void run(int id, AtomicInteger processed) {
          for (int w = 0; w < model.windowLengths.length; w++) {
            if (w % BLOCKS == id) {
              int[][] words = model.createWords(samples, w);
              WEASEL.BagOfBigrams[] bobForOneWindow = fitOneWindow(
                  samples,
                  model.windowLengths, norm,
                  words, maxF, w);

              mergeBobs(bop, bobForOneWindow);
            }
          }
        }
      });

      model.trainChiSquared(bop, chi);

      // train liblinear
      Problem problem = initLibLinearProblem(bop, model.dict, bias);
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));
      int correct = 0;
      for (int j = 0; j < problem.x.length; j++) {
        correct += Linear.predict(linearModel, problem.x[j])==problem.y[j] ? 1 : 0;
      }

      System.out.println("Train Dict Size: " + model.dict.size());

      // obtain feature weights
      //double[] weights = linearModel.getFeatureWeights();

      TimeSeries[] slidingTs = sample.getSubsequences(width, norm);

      double[] plot = new double[sample.getLength()];
      for (int w = 0; w < model.windowLengths.length; w++) {
        int windowLength = model.windowLengths[w];

        // left split
        int[][] words = model.createWords(slidingTs, w);
        WEASEL.BagOfBigrams[] bob = model.createBagOfPatterns(words, slidingTs, w, maxF);

        for (int i = 0; i < words.length; i++) {
          for (int j = 0; j < words[i].length; j++) {
            if (model.dict.dict.containsKey(words[i][j])) {
              int index = model.dict.getWordIndex(words[i][j]);
              //double weight = weights[index];
              //int count = bob[i].bob.get(words[i][j]);
              for (int w2 = 0; w2 < windowLength; w2++) {
                plot[i + j + w2] = Math.max(index, plot[i + j + w2]);
              }
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