package subwordTransformer.apriori;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import de.mrapp.apriori.Apriori;
import de.mrapp.apriori.Filter;
import de.mrapp.apriori.FrequentItemSets;
import de.mrapp.apriori.ItemSet;
import de.mrapp.apriori.Output;
import de.mrapp.apriori.Transaction;
import subwordTransformer.SubwordTransformer;

/**
 * A transformer that uses the Apriori algorithm to find frequent patterns
 * (character sets). Only works with positional alphabets.
 */
public class AprioriTransformer extends SubwordTransformer<AprioriParameter> {

  private FrequentItemSets<CharacterItem> frequentItemSets;
  private double currentMinSupport = 1;
  private List<short[]> dictionary;

  /**
   * @param alphabetSize the alphabet size of the input words
   */
  public AprioriTransformer(int alphabetSize) {
    super(alphabetSize, true); // AprioriTransformer only works with positional alphabets (SFA)
  }

  /**
   * @param alphabetSize  the alphabet size of the input words
   * @param fillCharacter the character to be used for wildcards (default: -1)
   */
  public AprioriTransformer(int alphabetSize, short fillCharacter) {
    super(alphabetSize, true, fillCharacter);
  }

  @Override
  public int getOutputAlphabetSize() {
    return this.getInputAlphabetSize() + 1;
  }

  @Override
  protected void buildDictionary() {
    currentMinSupport = this.getParameter().getMinSupport();
    List<Transaction<CharacterItem>> transactions = new ArrayList<>(this.getWords().length);
    for (short[] word : this.getWords()) {
      transactions.add(new CharacterTransaction(word, this.getInputAlphabetSize()));
    }
    Apriori<CharacterItem> apriori = new Apriori.Builder<CharacterItem>(currentMinSupport).create();
    Output<CharacterItem> output = apriori.execute(transactions);
    frequentItemSets = output.getFrequentItemSets();
    this.fillDictionary(frequentItemSets);
  }

  @Override
  protected void updateDictionary() {
    if (this.getParameter().getMinSupport() >= currentMinSupport) {
      @SuppressWarnings("rawtypes")
      Filter<ItemSet> filter = Filter.forItemSets().bySupport(this.getParameter().getMinSupport());
      FrequentItemSets<CharacterItem> filteredFrequentItemSets = frequentItemSets.filter(filter);
      this.fillDictionary(filteredFrequentItemSets);
    } else {
      this.buildDictionary();
    }
  }

  private void fillDictionary(FrequentItemSets<CharacterItem> frequentItemSets) {
    dictionary = new ArrayList<>();
    int alphabetSize = this.getInputAlphabetSize();
    for (ItemSet<CharacterItem> itemSet : frequentItemSets) {
      int patternLength = itemSet.last().getChar() / alphabetSize + 1;
      short[] pattern = new short[patternLength];
      Arrays.fill(pattern, this.getFillCharacter());
      for (CharacterItem item : itemSet) {
        short positionalChar = item.getChar();
        int position = positionalChar / alphabetSize;
        short character = (short) (positionalChar % alphabetSize);
        pattern[position] = character;
      }
      dictionary.add(pattern);
    }
  }

  @Override
  public short[][] transform(short[] word) {
    List<short[]> matchingSubwords = new ArrayList<>();
    short wildcard = this.getFillCharacter();
    for (short[] pattern : dictionary) {
      if (matchesWord(pattern, word, wildcard)) {
        short[] subword = new short[word.length];
        Arrays.fill(subword, this.getFillCharacter());
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

}
