package subwordTransformer.apriori;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import de.mrapp.apriori.Transaction;

class CharacterTransaction implements Transaction<CharacterItem> {

  private final List<CharacterItem> items;

  public CharacterTransaction(short[] word, int alphabetSize) {
    this.items = new ArrayList<CharacterItem>(word.length);
    for (int i = 0; i < word.length; i++) {
      this.items.add(new CharacterItem(word[i], i, alphabetSize));
    }
  }

  @Override
  public Iterator<CharacterItem> iterator() {
    return this.items.iterator();
  }

}
