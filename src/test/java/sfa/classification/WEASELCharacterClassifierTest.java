package sfa.classification;

import static junit.framework.TestCase.assertEquals;

import java.io.File;
import java.util.ArrayList;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.SFAWordsTest;
import sfa.classification.Classifier.Score;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import subwordTransformer.no.NoParameter;
import subwordTransformer.no.NoTransformer;

@RunWith(JUnit4.class)
public class WEASELCharacterClassifierTest {

  private static String[] datasets = new String[] { "Coffee", "Beef", "CBF", "variable_length" };

  @Test
  public void testTransformerIntegration() {

    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    File dir = new File(classLoader.getResource("datasets/univariate/").getFile());

    for (String s : datasets) {
      File d = new File(dir.getAbsolutePath() + "/" + s);
      Assert.assertTrue("Dataset could not be found: " + d.getAbsolutePath() + ".", d.exists() && d.isDirectory());
      boolean found = false;
      for (File train : d.listFiles()) {
        if (train.getName().toUpperCase().endsWith("TRAIN")) {
          File test = new File(train.getAbsolutePath().replaceFirst("TRAIN", "TEST"));

          Assert.assertTrue("File " + test.getName() + " does not exist", test.exists());

          // Load the train/test splits
          TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(test);
          TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(train);

          WEASELCharacterClassifier.transformer = new NoTransformer(WEASELCharacterClassifier.maxS);
          WEASELCharacterClassifier.transformerParameterList = new ArrayList<>(NoParameter.getParameterList());

          WEASELCharacterClassifier w1 = new WEASELCharacterClassifier();
          Score scoreW1 = w1.eval(trainSamples, testSamples);
          WEASELClassifier w2 = new WEASELClassifier();
          Score scoreW2 = w2.eval(trainSamples, testSamples);

          assertEquals("Training accuracy differs for dataset " + s + ".", scoreW2.getTrainingAccuracy(), scoreW1.getTrainingAccuracy(), 2.0 / trainSamples.length);
          assertEquals("Testing accuracy differs for dataset " + s + ".", scoreW2.getTestingAccuracy(), scoreW1.getTestingAccuracy(), 2.0 / testSamples.length);

          found = true;
          break;
        }
      }

      Assert.assertTrue("Train file could not be found in " + d.getAbsolutePath() + ".", found);

    }
  }

}
