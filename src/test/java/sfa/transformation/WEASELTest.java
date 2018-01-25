package sfa.transformation;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

@RunWith(JUnit4.class)
public class WEASELTest {

  /**
   * Test to create words with window sizes larger than the time series length
   * @throws IOException
   */
  @Test
  public void testCreateWordsFromVariableLengthTS() throws IOException {

    Random r = new Random(1);
    ArrayList<TimeSeries> samples = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
      TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(1+i, r);
      samples.add(timeSeries);
    }

    int[] windowLengths = new int[]{1024,128,64,32,4};
    WEASEL model = new WEASEL(12, 8, windowLengths, false, true);
    int[][][] words = model.createWords(samples.toArray(new TimeSeries[]{}));
    for (int i = 0; i < words.length; i++) {
      Assert.assertNotNull("Index is null: " +i, words[i]);
      //System.out.println("i:" + words.length);
      for (int j = 0; j < words[i].length; j++) {
        Assert.assertNotNull("Index is null: " +i+ ":" + j, words[i][j]);
        //System.out.println("j:" + words[i].length);
        for (int k = 0; k < words[i][j].length; k++) {
          Assert.assertNotNull("Index is null: " +i+ ":" + j + ":" + k, words[i][j][k]);
          //System.out.println("k:" + words[i][j].length);
        }
      }
    }

    WEASEL.BagOfBigrams[] bag = model.createBagOfPatterns(words, samples.toArray(new TimeSeries[]{}), 8);
    Assert.assertNotNull("BagOfBigrams is null:", bag);

    // create bag of words from a time series without data
    TimeSeries ts = new TimeSeries(new double[]{});
    int[][][] words2 = model.createWords(new TimeSeries[]{ts});
    Assert.assertNotNull("Word is null:", words2[0]);
    WEASEL.BagOfBigrams[] bag2 = model.createBagOfPatterns(words2, new TimeSeries[]{ts}, 8);
    Assert.assertNotNull("BagOfBigrams is null:", bag2);

    System.out.println("All done");
  }

  /**
   * Create bag of words from a time series without data
   * @throws IOException
   */
  @Test
  public void testCreateWordsFromEmptyTS() throws IOException {
    int[] windowLengths = new int[]{1024,128,64,32,4};
    WEASEL model = new WEASEL(12, 8, windowLengths, false, true);
    TimeSeries ts = new TimeSeries(new double[]{});
    int[][][] words2 = model.createWords(new TimeSeries[]{ts});
    Assert.assertNotNull("Word is null:", words2[0]);
    WEASEL.BagOfBigrams[] bag2 = model.createBagOfPatterns(words2, new TimeSeries[]{ts}, 8);
    Assert.assertNotNull("BagOfBigrams is null:", bag2);
  }
}

