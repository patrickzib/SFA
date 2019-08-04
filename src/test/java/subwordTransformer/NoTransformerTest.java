package subwordTransformer;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import subwordTransformer.no.NoParameter;
import subwordTransformer.no.NoTransformer;

@RunWith(JUnit4.class)
public class NoTransformerTest extends TransformerTest {

  @Test
  public void testTransform() {

    Random r = new Random(1);

    int alphabetSize = 4;
    NoTransformer tr = new NoTransformer(alphabetSize);
    short[][] trainingWords = new short[10][];
    for (int i = 0; i < 10; i++) {
      int wordLength = r.nextInt(6) + 1;
      trainingWords[i] = new short[wordLength];
      for (int j = 0; j < wordLength; j++) {
        trainingWords[i][j] = (short) r.nextInt(alphabetSize);
      }
    }
    tr.setWords(trainingWords);
    tr.setParameter(new NoParameter());
    tr.fit();

    for (int i = 0; i < 100; i++) {
      int wordLength = r.nextInt(6) + 1;
      short[] word = new short[wordLength];
      for (int j = 0; j < wordLength; j++) {
        word[j] = (short) r.nextInt(alphabetSize);
      }
      short[][] subwords = tr.transformWord(word);
      Assert.assertArrayEquals("NoTransformer does NOT output nothing", new short[][] {}, subwords);
    }

  }
}
