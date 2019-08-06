package subwordTransformer.bpe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

class BPEUtils {
  static List<List<Short>> getWordList(short[] word, boolean positionalAlphabets, int inputAlphabetSize) {
    List<List<Short>> wordList = new ArrayList<>(word.length);
    for (int j = 0; j < word.length; j++) {
      List<Short> charList = new ArrayList<>(1);
      short character;
      if (positionalAlphabets) {
        character = (short) (word[j] + inputAlphabetSize * j);
      } else {
        character = word[j];
      }
      charList.add(character);
      wordList.add(charList);
    }
    return wordList;
  }

  static Map<List<List<Short>>, Integer> buildVocab(short[][] words, boolean positionalAlphabets, int inputAlphabetSize) {
    Map<List<List<Short>>, Integer> vocab = new HashMap<>();
    for (short[] word : words) {
      List<List<Short>> wordList = getWordList(word, positionalAlphabets, inputAlphabetSize);
      if (vocab.containsKey(wordList)) {
        vocab.put(wordList, vocab.get(wordList) + 1);
      } else {
        vocab.put(wordList, 1);
      }
    }
    return vocab;
  }

  static Map<List<List<Short>>, Integer> getStats(Map<List<List<Short>>, Integer> vocab) {
    Map<List<List<Short>>, Integer> pairs = new HashMap<>();
    for (Entry<List<List<Short>>, Integer> wordEntry : vocab.entrySet()) {
      List<List<Short>> word = wordEntry.getKey();
      for (int i = 0; i < word.size() - 1; i++) {
        List<List<Short>> pairList = new ArrayList<>(2);
        pairList.add(word.get(i));
        pairList.add(word.get(i + 1));
        if (pairs.containsKey(pairList)) {
          pairs.put(pairList, pairs.get(pairList) + wordEntry.getValue());
        } else {
          pairs.put(pairList, wordEntry.getValue());
        }
      }
    }
    return pairs;
  }

  static Map<List<List<Short>>, Integer> mergeVocab(Map<List<List<Short>>, Integer> vocab, List<List<Short>> pair, List<Short> mergedPair) {
    Map<List<List<Short>>, Integer> newVocab = new HashMap<>(vocab.size());
    for (Entry<List<List<Short>>, Integer> wordEntry : vocab.entrySet()) {
      List<List<Short>> word = mergeWord(wordEntry.getKey(), pair, mergedPair);
      newVocab.put(word, wordEntry.getValue());
    }
    return newVocab;
  }

  private static List<List<Short>> mergeWord(List<List<Short>> word, List<List<Short>> pair, List<Short> mergedPair) {
    for (int i = 0; i < word.size() - 1; i++) {
      if (word.get(i).equals(pair.get(0)) && word.get(i + 1).equals(pair.get(1))) {
        word.set(i, mergedPair);
        word.remove(i + 1);
      }
    }
    return word;
  }

  static short[][] transform(short[] word, List<List<List<Short>>> merges, boolean positionalAlphabets, int inputAlphabetSize, short fillCharacter) {
    // apply merges
    List<List<Short>> wordList = getWordList(word, positionalAlphabets, inputAlphabetSize);
    for (List<List<Short>> pair : merges) {
      List<Short> mergedPair = getMergedPair(pair);
      wordList = mergeWord(wordList, pair, mergedPair);
    }
    // return merged subwords
    List<short[]> subwords = new ArrayList<>();
    for (List<Short> subword : wordList) {
      if (subword.size() > 1) {
        if (positionalAlphabets) {
          subwords.add(shortListToArrayPattern(subword, word.length, fillCharacter, inputAlphabetSize));
        } else {
          subwords.add(shortListToArray(subword));
        }
      }
    }
    return subwords.toArray(new short[subwords.size()][]);
  }

  private static short[] shortListToArray(List<Short> word) {
    short[] wordArray = new short[word.size()];
    for (int i = 0; i < word.size(); i++) {
      wordArray[i] = word.get(i);
    }
    return wordArray;
  }

  private static short[] shortListToArrayPattern(List<Short> word, int length, short fillCharacter, int inputAlphabetSize) {
    short[] pattern = new short[length];
    Arrays.fill(pattern, fillCharacter);
    int alphabetSize = inputAlphabetSize;
    for (short s : word) {
      int position = s / alphabetSize;
      short character = (short) (s % alphabetSize);
      pattern[position] = character;
    }
    return pattern;
  }

  static List<Short> getMergedPair(List<List<Short>> pair) {
    List<Short> mergedPair = new ArrayList<>(pair.get(0).size() + pair.get(1).size());
    mergedPair.addAll(pair.get(0));
    mergedPair.addAll(pair.get(1));
    return mergedPair;
  }
}
