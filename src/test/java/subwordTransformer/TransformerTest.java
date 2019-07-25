package subwordTransformer;

import java.util.Arrays;
import java.util.Comparator;

import org.junit.Assert;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public abstract class TransformerTest {

  protected static void assertArrayEqualAnyOrder(String msg, short[][] expecteds, short[][] actuals) {
    sortWordArray(expecteds);
    sortWordArray(actuals);
    Assert.assertArrayEquals(msg, expecteds, actuals);
  }

  protected static void sortWordArray(short[][] words) {
    Arrays.sort(words, new Comparator<short[]>() {
      @Override
      public int compare(final short[] word1, final short[] word2) {
        if (word1.length < word2.length)
          return -1;
        if (word1.length > word2.length)
          return 1;
        for (int pos = 0; pos < word1.length; pos++) {
          if (word1[pos] < word2[pos])
            return -1;
          if (word1[pos] > word2[pos])
            return 1;
        }
        return 0;
      }
    });
  }

}
