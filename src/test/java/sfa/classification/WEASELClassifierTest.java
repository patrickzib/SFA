package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class WEASELClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //variable_length;Weasel;0.929;0.964
        dataSets.add(new DataSet("variable_length", 0.929, 0.964));
        //Coffee;Weasel;1.0;1.0
        dataSets.add(new DataSet("Coffee", 1.0, 1.0));
        //Beef;Weasel;0.933;0.867
        dataSets.add(new DataSet("Beef", 0.933, 0.867));
        //CBF;Weasel;0.967;0.996
        dataSets.add(new DataSet("CBF", 0.967, 0.996));

        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        return new WEASELClassifier();
    }


}

