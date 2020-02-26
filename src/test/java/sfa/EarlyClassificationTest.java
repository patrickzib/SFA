// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.classification.*;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.File;
import java.io.IOException;

@RunWith(JUnit4.class)
public class EarlyClassificationTest {

  // The datasets to use
  public static String[] datasets = new String[]{
//          "Coffee",
//          "CBF",
//          "Beef",
//          "ECG200", "FaceFour", "OliveOil",
//          "Gun_Point",
//          "DiatomSizeReduction",
//          "ECGFiveDays",
//          "TwoLeadECG",
//          "SonyAIBORobot SurfaceII",
//          "MoteStrain",
//          "ItalyPowerDemand",
//          "SonyAIBORobot Surface",

      "Two_Patterns",
      "ChlorineConcentration",
      "wafer",
      "MedicalImages",
      "FaceAll",
      "OSULeaf",
      "Adiac",
      "SwedishLeaf",
      "yoga",
      "fish",
      "Lighting7",
      "Lighting2",
      "Trace",
      "synthetic_control",
      "FacesUCR",
      "CinC_ECG_torso",
      "MALLAT",
      "Symbols",

      "Coffee",
      "ECG200",
      "FaceFour",
      "OliveOil",
      "Gun_Point",
      "Beef",
      "DiatomSizeReduction",
      "CBF",
      "ECGFiveDays",
      "TwoLeadECG",
      "SonyAIBORobot SurfaceII",
      "MoteStrain",
      "ItalyPowerDemand",
      "SonyAIBORobot Surface",

      "Haptics",
      "InlineSkate",
      "50words",
      "Cricket_Y",
      "Cricket_X",
      "Cricket_Z",
      "WordSynonyms",
      "uWaveGestureLibrary_Z",
      "uWaveGestureLibrary_Y",
      "uWaveGestureLibrary_X",

      "NonInvasiveFatalECG_Thorax1",
      "NonInvasiveFatalECG_Thorax2",
      "StarLightCurves",
      "Car",
      "Plane",
      "ArrowHead",
      "BeetleFly",
      "BirdChicken",
      "Computers",
      "DistalPhalanxOutlineAgeGroup",
      "DistalPhalanxOutlineCorrect",
      "DistalPhalanxTW",
      "Earthquakes",
      "ECG5000",
      "ElectricDevices",
      "FordA",
      "FordB",
      "Ham",
      "HandOutlines",
      "Herring",
      "InsectWingbeatSound",
      "LargeKitchenAppliances",
      "Meat",
      "MiddlePhalanxOutlineAgeGroup",
      "MiddlePhalanxOutlineCorrect",
      "MiddlePhalanxTW",
      "PhalangesOutlinesCorrect",
      "Phoneme",
      "ProximalPhalanxOutlineAgeGroup",
      "ProximalPhalanxOutlineCorrect",
      "ProximalPhalanxTW",
      "RefrigerationDevices",
      "ScreenType",
      "ShapeletSim",
      "ShapesAll",
      "SmallKitchenAppliances",
      "Strawberry",
      "ToeSegmentation1",
      "ToeSegmentation2",
      "UWaveGestureLibraryAll",
      "Wine",
      "Worms",
      "WormsTwoClass"
  };

  @Test
  public void testUCRClassification() throws IOException {
    try {
      // the relative path to the datasets
      ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

      //File dir = new File(classLoader.getResource("datasets/univariate/").getFile());
      //File dir = new File("/Users/bzcschae/workspace/similarity/datasets/classification");
      File dir = new File("/vol/fob-wbib-vol2/wbi/schaefpa/similarity/classification");

      TimeSeries.APPLY_Z_NORM = false;

      for (String s : datasets) {
        File d = new File(dir.getAbsolutePath() + "/" + s);
        if (d.exists() && d.isDirectory()) {
          for (File train : d.listFiles()) {
            //for (double p_value : new double[]{0.5,0.2,0.1,0.05,0.01}) {
              if (train.getName().toUpperCase().endsWith("TRAIN")) {
                File test = new File(train.getAbsolutePath().replaceFirst("TRAIN", "TEST"));

                if (!test.exists()) {
                  System.err.println("File " + test.getName() + " does not exist");
                  test = null;
                }

                Classifier.DEBUG = false;

                // Load the train/test splits
                TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(test);
                TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(train);

                // The TEASER-classifier
                TEASERClassifier t = new TEASERClassifier();
                TEASERClassifier.S = 20.0;

                //WEASELClassifier.p_value = p_value;

                Classifier.Score scoreT = t.eval(trainSamples, testSamples);
                System.out.println(s + ";" + scoreT.toString());

              }
            //}
          }
        } else {
          // not really an error. just a hint:
          System.out.println("Dataset could not be found: " + d.getAbsolutePath() + ". " +
                  "Please download datasets from [http://www.cs.ucr.edu/~eamonn/time_series_data/].");
        }
      }
    } finally {
      TimeSeries.APPLY_Z_NORM = true; // FIXME static variable breaks some test cases!
    }
  }

  public static void main(String[] args) throws IOException {
    EarlyClassificationTest ucr = new EarlyClassificationTest();
    ucr.testUCRClassification();
  }
}
