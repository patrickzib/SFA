package subwordTransformer;

import java.util.Arrays;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import subwordTransformer.cng.CNGParameter;
import subwordTransformer.cng.CNGTransformer;
import subwordTransformer.cng.SupervisedCNGTransformer;

@RunWith(JUnit4.class)
public class CNGTransformerTest extends TransformerTest {

  @Test
  public void testTransformNoPositionalAlphabet() {

    CNGTransformer tr = new CNGTransformer(4, false);

    short[][] trainingWords = new short[][] { { 2, 0, 1, 1 }, { 0, 0, 1, 2 }, { 3, 2, 0, 0 }, { 3, 0, 3, 2 }, { 3, 3, 3, 1 }, { 1, 3, 1, 0 }, { 1, 1, 2, 3 }, { 0, 2, 3, 2 }, { 2, 0, 0, 3 },
        { 1, 2, 0, 3 } };
    tr.setWords(trainingWords);

    short[][] words = new short[][] { { 0, 2, 0, 0 }, { 2, 0, 3, 1 }, { 3, 2, 0, 0 }, { 0, 1, 2, 3 } };
    short[][][] expecteds = new short[][][] { { { 0, 0 }, { 2, 0, 0 }, { 2, 0 } }, { { 0, 3 }, { 2, 0 } }, { { 2, 0 } }, {} };
    CNGParameter[] parameters = new CNGParameter[] { new CNGParameter(2, 3, 0.2), new CNGParameter(2, 3, 0.3), new CNGParameter(2, 3, 0.4), new CNGParameter(2, 3, 0.5) };

    for (int i = 0; i < words.length; i++) {
      tr.setParameter(parameters[i]);
      tr.fit();
      short[][] subwords = tr.transformWord(words[i]);
      assertArrayEqualAnyOrder("transformation result of " + Arrays.toString(words[i]) + " with " + parameters[i] + " does NOT match", expecteds[i], subwords);
    }

  }

  @Test
  public void testTransformWithPositionalAlphabet() {

    CNGTransformer tr = new CNGTransformer(4, true, (short) -1);

    short[][] trainingWords = new short[][] { { 2, 0, 1, 1 }, { 0, 0, 1, 2 }, { 3, 2, 0, 0 }, { 3, 0, 3, 2 }, { 3, 3, 3, 1 }, { 1, 3, 1, 0 }, { 1, 1, 2, 3 }, { 0, 2, 3, 2 }, { 2, 0, 0, 3 },
        { 1, 2, 0, 3 } };
    tr.setWords(trainingWords);

    short[][] words = new short[][] { { 2, 0, 1, 3 }, { 2, 0, 1, 0 }, { 0, 1, 2, 3 } };
    short[][][] expecteds = new short[][][] { { { 2, 0, -1, -1 }, { -1, 0, 1, -1 }, { 2, 0, 1, -1 } }, { { 2, 0, -1, -1 }, { -1, 0, 1, -1 } }, {} };
    CNGParameter[] parameters = new CNGParameter[] { new CNGParameter(2, 3, 0.1), new CNGParameter(2, 3, 0.2), new CNGParameter(2, 3, 0.3) };

    for (int i = 0; i < words.length; i++) {
      tr.setParameter(parameters[i]);
      tr.fit();
      short[][] subwords = tr.transformWord(words[i]);
      assertArrayEqualAnyOrder("transformation result of " + Arrays.toString(words[i]) + " with " + parameters[i] + " does NOT match", expecteds[i], subwords);
    }

  }

  @Test
  public void testSupervisedTransform() {

    SupervisedCNGTransformer tr = new SupervisedCNGTransformer(4, true, (short) -1);

    short[][][] trainingWords = new short[][][] { { { 2, 0, 1, 1 }, { 0, 0, 1, 2 }, { 3, 2, 0, 0 }, { 3, 0, 3, 2 }, { 3, 3, 3, 1 }, { 1, 3, 1, 0 } },
        { { 1, 1, 2, 3 }, { 0, 2, 3, 2 }, { 2, 0, 0, 3 }, { 1, 2, 0, 3 } } };
    tr.setWords(trainingWords);

    short[][] words = new short[][] { { 2, 0, 1, 3 }, { 2, 0, 1, 0 } };
    short[][][] expecteds = new short[][][] { { { -1, 0, 1, -1 }, { 2, 0, 1, -1 } }, { { 2, 0, 1, -1 }, { -1, 0, 1, -1 }, { -1, -1, 1, 0 } } };
    CNGParameter[] parameters = new CNGParameter[] { new CNGParameter(2, 3, 0.5), new CNGParameter(2, 3, 1.0) };

    for (int i = 0; i < words.length; i++) {
      tr.setParameter(parameters[i]);
      tr.fit();
      short[][] subwords = tr.transformWord(words[i]);
      assertArrayEqualAnyOrder("transformation result of " + Arrays.toString(words[i]) + " with " + parameters[i] + " does NOT match", expecteds[i], subwords);
    }

  }

}
