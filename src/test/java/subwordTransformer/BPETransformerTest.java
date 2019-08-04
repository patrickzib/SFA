package subwordTransformer;

import java.util.Arrays;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import subwordTransformer.bpe.BPEParameter;
import subwordTransformer.bpe.BPETransformer;

@RunWith(JUnit4.class)
public class BPETransformerTest extends TransformerTest {

  @Test
  public void testTransformNoPositionalAlphabet() {

    BPETransformer tr = new BPETransformer(4, false);

    short[][] trainingWords = new short[][] { { 2, 0, 1, 1 }, { 0, 0, 1, 2 }, { 3, 2, 0, 0 }, { 3, 0, 3, 2 }, { 3, 3, 3, 1 }, { 1, 3, 1, 0 }, { 1, 1, 2, 3 }, { 0, 2, 3, 2 }, { 2, 0, 0, 3 },
        { 1, 2, 0, 3 } };
    tr.setWords(trainingWords);

    short[][] words = new short[][] { { 0, 1, 2, 3 }, { 3, 1, 2, 0 }, { 2, 0, 0, 0 } };
    short[][][] expecteds = new short[][][] { {}, { { 2, 0 } }, { { 2, 0, 0 } } };
    BPEParameter[] parameters = new BPEParameter[] { new BPEParameter(0.5), new BPEParameter(0.3), new BPEParameter(0.2) };

    for (int i = 0; i < words.length; i++) {
      tr.setParameter(parameters[i]);
      tr.fit();
      short[][] subwords = tr.transformWord(words[i]);
      assertArrayEqualAnyOrder("transformation result of " + Arrays.toString(words[i]) + " with " + parameters[i] + " does NOT match", expecteds[i], subwords);
    }

  }

  @Test
  public void testTransformWithPositionalAlphabet() {

    BPETransformer tr = new BPETransformer(4, true, (short) -1);

    short[][] trainingWords = new short[][] { { 2, 0, 1, 1 }, { 0, 0, 1, 2 }, { 3, 2, 0, 0 }, { 3, 0, 3, 2 }, { 3, 3, 3, 1 }, { 1, 3, 1, 0 }, { 1, 1, 2, 3 }, { 0, 2, 3, 2 }, { 2, 0, 0, 3 },
        { 1, 2, 0, 3 } };
    tr.setWords(trainingWords);

    short[][] words = new short[][] { { 0, 1, 2, 3 }, { 2, 0, 1, 3 } };
    short[][][] expecteds = new short[][][] { {}, { { 2, 0, -1, -1 } } };
    BPEParameter[] parameters = new BPEParameter[] { new BPEParameter(0.3), new BPEParameter(0.2) };

    for (int i = 0; i < words.length; i++) {
      tr.setParameter(parameters[i]);
      tr.fit();
      short[][] subwords = tr.transformWord(words[i]);
      assertArrayEqualAnyOrder("transformation result of " + Arrays.toString(words[i]) + " with " + parameters[i] + " does NOT match", expecteds[i], subwords);
    }

  }

}
