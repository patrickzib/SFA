package subwordTransformer.cng;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class CNGUtils {
  static Map<List<Short>, Integer> countNGrams(int n, short[][] words, boolean positionalAlphabets, short fillCharacter) {
    Map<List<Short>, Integer> countMap = new HashMap<>();
    for (short[] word : words) {
      int lastIndex = word.length - n;
      for (int i = 0; i <= lastIndex; i++) {
        List<Short> subword;
        if (positionalAlphabets) {
          subword = shortArrayToListPattern(word, i, n, fillCharacter);
        } else {
          subword = shortArrayToList(word, i, n);
        }
        if (countMap.containsKey(subword)) {
          countMap.put(subword, countMap.get(subword) + 1);
        } else {
          countMap.put(subword, 1);
        }
      }
    }
    return countMap;
  }

  private static List<Short> shortArrayToList(short[] word, int startIndex, int length) {
    List<Short> wordList = new ArrayList<>();
    for (int i = startIndex; i < startIndex + length; i++) {
      wordList.add(word[i]);
    }
    return wordList;
  }

  private static List<Short> shortArrayToListPattern(short[] word, int startIndex, int length, short wildcard) {
    List<Short> wordList = new ArrayList<>();
    for (int i = 0; i < startIndex; i++) {
      wordList.add(wildcard);
    }
    wordList.addAll(shortArrayToList(word, startIndex, length));
    for (int i = startIndex + length; i < word.length; i++) {
      wordList.add(wildcard);
    }
    return wordList;
  }

  static short[] shortListToArray(List<Short> word) {
    short[] wordArray = new short[word.size()];
    for (int i = 0; i < word.size(); i++) {
      wordArray[i] = word.get(i);
    }
    return wordArray;
  }

  private static boolean isSubArray(short[] subArray, short[] array) {
    int subArrayLength = subArray.length;
    int arrayLength = array.length;
    boolean contained = false;
    for (int i = 0; i <= arrayLength - subArrayLength; i++) {
      boolean found = true;
      for (int j = 0; j < subArrayLength; j++) {
        if (subArray[j] != array[i + j]) {
          found = false;
          break;
        }
      }
      if (found) {
        contained = true;
        break;
      }
    }
    return contained;
  }

  private static boolean matchesWord(short[] pattern, short[] word, short wildcard) {
    if (pattern.length != word.length) {
      return false;
    }
    for (int i = 0; i < pattern.length; i++) {
      if (pattern[i] != wildcard && pattern[i] != word[i]) {
        return false;
      }
    }
    return true;
  }

  static short[][] transform(short[] word, List<short[]> dictionary, boolean positionalAlphabets, short fillCharacter) {
    List<short[]> matchingSubwords = new ArrayList<>();
    for (short[] subword : dictionary) {
      if ((positionalAlphabets && matchesWord(subword, word, fillCharacter)) || (!positionalAlphabets && isSubArray(subword, word))) {
        matchingSubwords.add(subword);
      }
    }
    return matchingSubwords.toArray(new short[matchingSubwords.size()][]);
  }
}
