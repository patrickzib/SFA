package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class TEASERClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {

        // these numbers are with z-normalization

        List<DataSet> dataSets=new ArrayList<>();
        //variable_length;TEASER;
        dataSets.add(new DataSet("variable_length", 0.964, 0.929, 0.333));
        //Coffee;TEASER;
        dataSets.add(new DataSet("Coffee", 1.0, 1.0, 0.47));
        //CBF;TEASER;
        dataSets.add(new DataSet("CBF", 1.0, 0.982, 0.64));
        //Beef;TEASER;
        //takes too long to compute, dataSets.add(new DataSet("Beef", 0.933, 0.867, 0.46));

        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        TEASERClassifier teaser = new TEASERClassifier();
        TEASERClassifier.S = 10; // faster processing
        return teaser;
    }


}

