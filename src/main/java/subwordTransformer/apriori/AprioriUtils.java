package subwordTransformer.apriori;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import de.mrapp.apriori.Apriori;
import de.mrapp.apriori.FrequentItemSets;
import de.mrapp.apriori.ItemSet;
import de.mrapp.apriori.Output;
import de.mrapp.apriori.Transaction;

class AprioriUtils {
  static short[][] transform(short[] word, List<short[]> dictionary, short fillCharacter) {
    List<short[]> matchingSubwords = new ArrayList<>();
    for (short[] pattern : dictionary) {
      if (matchesWord(pattern, word, fillCharacter)) {
        short[] subword = new short[word.length];
        Arrays.fill(subword, fillCharacter);
        System.arraycopy(pattern, 0, subword, 0, pattern.length);
        matchingSubwords.add(subword);
      }
    }
    return matchingSubwords.toArray(new short[matchingSubwords.size()][]);
  }

  private static boolean matchesWord(short[] pattern, short[] word, short wildcard) {
    if (pattern.length > word.length) {
      return false;
    }
    for (int i = 0; i < pattern.length; i++) {
      if (pattern[i] != wildcard && pattern[i] != word[i]) {
        return false;
      }
    }
    return true;
  }

  static FrequentItemSets<CharacterItem> getFrequentItemSets(short[][] words, double minSupport, int inputAlphabetSize) {
    List<Transaction<CharacterItem>> transactions = new ArrayList<>(words.length);
    for (short[] word : words) {
      transactions.add(new CharacterTransaction(word, inputAlphabetSize));
    }
    Apriori<CharacterItem> apriori = new Apriori.Builder<CharacterItem>(minSupport).create();
    Output<CharacterItem> output = apriori.execute(transactions);
    return output.getFrequentItemSets();
  }

  static short[] itemSetToPattern(ItemSet<CharacterItem> itemSet, short fillCharacter) {
    int patternLength = itemSet.last().getPosition() + 1;
    short[] pattern = new short[patternLength];
    Arrays.fill(pattern, fillCharacter);
    for (CharacterItem item : itemSet) {
      pattern[item.getPosition()] = item.getCharacter();
    }
    return pattern;
  }

  static List<Short> itemSetToPatternList(ItemSet<CharacterItem> itemSet, short fillCharacter) {
    List<Short> list = new ArrayList<>();
    for (short character : itemSetToPattern(itemSet, fillCharacter)) {
      list.add(character);
    }
    return list;
  }

  static short[] shortListToArray(List<Short> word) {
    short[] wordArray = new short[word.size()];
    for (int i = 0; i < word.size(); i++) {
      wordArray[i] = word.get(i);
    }
    return wordArray;
  }

}
