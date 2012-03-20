/*
 * Copyright 2010
 * This file is part of XXXXXX.
 * XXXXXX is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 * 
 * XXXXXX is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * XXXXXX; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package alvs.modules.simulation;

import alvs.data.BugDataset;
import alvs.data.PeakListRow;
import alvs.util.Range;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.ComplementNaiveBayes;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiScheme;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.meta.Stacking;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.lmt.LogisticBase;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 *
 * @author bicha
 */
public class Bug {

        private Cell cell;
        private int x, y;
        private List<PeakListRow> rowList;
        private int life = 300;
        private BugDataset dataset;
        private Classifier classifier;
        private classifiersEnum classifierType;
        private double wellClassified, total;
        private double sensitivity = 0;
        private double specificity = 0;
        double spec = 0, sen = 0, totalspec = 0, totalsen = 0;
        private Random rand;
        private int MAXNUMBERGENES;
        Evaluation eval;
        boolean fixValue = false;
        Range range;
        int[] clusters;
        Instances training, test;
        double fValue = 0;
        double ridge = 0;

        public Bug(int x, int y, Cell cell, PeakListRow row, BugDataset dataset, int bugLife, int maxVariable, classifiersEnum classifier) {
                rand = new Random();
                this.cell = cell;
                this.range = cell.getRange();
                this.dataset = dataset;
                this.x = x;
                this.y = y;
                this.rowList = new ArrayList<PeakListRow>();
                if (row != null) {
                        this.rowList.add(row);
                }
                this.MAXNUMBERGENES = maxVariable;

                this.classifierType = classifiersEnum.LinearRegression;
                this.ridge = (double) (this.rand.nextInt(200) + 1) / 10;
                this.classify(cell.getRange());
                this.life = bugLife;

                clusters = new int[rowList.size()];
                for (int i = 0; i < rowList.size(); i++) {
                        clusters[i] = rowList.get(i).getCluster();
                }
        }

        public double getRidge() {
                return ridge;
        }

        public void eval() {
                try {                       
                        eval = new Evaluation(training);
                        eval.evaluateModel(classifier, training);
                       // eval.crossValidateModel(this.classifier, training, 10, rand);
                        this.fValue = eval.correlationCoefficient();
                } catch (Exception e) {
                        e.printStackTrace();
                }
        }

        @Override
        public Bug clone() {
                Bug newBug = new Bug(this.x, this.y, this.cell, null, this.dataset, this.life, this.MAXNUMBERGENES, this.classifierType);
                newBug.rowList = this.getRows();
                return newBug;
        }

        public double getAge() {
                return this.total;
        }

        public Bug(Bug father, Bug mother, BugDataset dataset, int bugLife, int maxVariable) {
                rand = new Random();
                this.dataset = dataset;
                this.cell = father.getCell();
                this.range = cell.getRange();
                this.x = father.getx();
                this.y = father.gety();
                this.rowList = new ArrayList<PeakListRow>();
                this.MAXNUMBERGENES = maxVariable;
                this.assingGenes(mother, false);
                this.assingGenes(father, false);

                this.orderPurgeGenes();


                this.classifierType = classifiersEnum.LinearRegression;
                if (father.getFMeasure() > mother.getFMeasure()) {
                        this.ridge = father.getRidge();
                } else {
                        this.ridge = mother.getRidge();
                }
                this.classify(cell.getRange());

                this.life = bugLife;

                clusters = new int[rowList.size()];
                for (int i = 0; i < rowList.size(); i++) {
                        clusters[i] = rowList.get(i).getCluster();
                }

        }

        public void setMaxVariable(int maxVariable) {
                this.MAXNUMBERGENES = maxVariable;
        }

        private void assingGenes(Bug parent, boolean isFather) {
                for (PeakListRow row : parent.getRows()) {
                        if (!this.rowList.contains(row)) {
                                this.rowList.add(row);
                                if (isFather) {
                                        break;
                                }
                        }
                }
        }

        public void orderPurgeGenes() {
                int removeGenes = this.rowList.size() - this.MAXNUMBERGENES;
                if (removeGenes > 0) {
                        for (int i = 0; i < removeGenes; i++) {
                                int index = getRepeatIndex();
                                if (index == -1) {
                                        index = rand.nextInt(this.rowList.size() - 1);
                                }
                                this.rowList.remove(index);
                        }
                }
        }

        public int getRepeatIndex() {
                for (int i = 0; i < this.rowList.size(); i++) {
                        PeakListRow row = this.rowList.get(i);
                        for (PeakListRow r : this.rowList) {
                                if (row != r && row.getCluster() == r.getCluster()) {
                                        return i;
                                }
                        }
                }
                return -1;
        }

        public classifiersEnum getClassifierType() {
                return this.classifierType;
        }

        /* public double getTotal() {
        return this.wellClassified / this.total;
        }

        /*  public double getSensitivity() {
        if ((Double.isNaN(this.sensitivity))) {
        return 0.0;
        }
        return this.sensitivity;
        }

        public double getSpecificity() {
        if ((Double.isNaN(this.specificity))) {
        return 0.0;
        }
        return this.specificity;
        }*/
        public double getFMeasure() {
                /* double value = (2 * this.sensitivity * this.specificity) / (this.sensitivity + this.specificity);
                if (Double.isNaN(value)) {
                return 0.0;
                } else {
                return value;
                }*/
                return fValue;
        }

        public List<PeakListRow> getRows() {
                return this.rowList;
        }

        public void setPosition(int x, int y) {
                this.x = x;
                this.y = y;
        }

        public int getx() {
                return this.x;
        }

        public int gety() {
                return this.y;
        }

        public void setCell(Cell cell) {
                this.cell = cell;
        }

        public Cell getCell() {
                return this.cell;
        }

        public double getLife() {
                return life;
        }

        public BugDataset getDataset() {
                return this.dataset;
        }

        boolean isDead() {
                life--;
                if (this.rowList.isEmpty()) {
                        life = 0;
                }
                if (this.life < 1 || this.life == Double.NaN) {
                        return true;
                } else {
                        return false;
                }
        }

        public void classify(Range range) {
                try {
                        getWekaDataset(range);
                        classifier = setClassifier();
                        if (classifier != null) {
                                classifier.buildClassifier(training);
                        }
                        ((LinearRegression) classifier).setRidge(this.ridge);
                } catch (Exception ex) {
                        ex.printStackTrace();
                        //Logger.getLogger(Bug.class.getName()).log(Level.SEVERE, null, ex);
                }
        }

        public void eat() {
                double prediction = isClassify();
                double difference = Math.abs(Double.parseDouble(cell.type) - prediction);
                this.life += (1 / difference);
                this.total++;

        }

        public double isClassify() {
                String sampleName = this.cell.getSampleName();
                FastVector attributes = new FastVector();
                for (int i = 0; i < rowList.size(); i++) {
                        Attribute weight = new Attribute("weight" + i);
                        attributes.addElement(weight);
                }

                Attribute type = new Attribute("class");
                attributes.addElement(type);
                //Creates the dataset
                Instances cellDataset = new Instances("Train Dataset", attributes, 0);
                double[] values = new double[cellDataset.numAttributes()];
                int cont = 0;
                for (PeakListRow row : rowList) {
                        values[cont++] = (Double) row.getPeak(sampleName);
                }
                values[cont] = Double.parseDouble(this.dataset.getSampleType(sampleName));
                Instance inst = new SparseInstance(1.0, values);
                cellDataset.add(inst);
                cellDataset.setClassIndex(cont);
                try {
                        return classifier.classifyInstance(cellDataset.instance(0));
                } catch (Exception ex) {
                        ex.printStackTrace();
                }
                return 1000;
        }

        public void increaseEnergy() {
                this.life += 0.5;
        }

        public void kill() {
                this.life = -1;
        }

        public void addLife() {
                this.life++;
        }        

        private Classifier setClassifier() {
                switch (this.classifierType) {
                        case Logistic:
                                return new Logistic();
                        case LogisticBase:
                                return new LogisticBase();
                        case LogitBoost:
                                return new LogitBoost();
                        case NaiveBayesMultinomialUpdateable:
                                return new NaiveBayesMultinomialUpdateable();
                        case NaiveBayesUpdateable:
                                return new NaiveBayesUpdateable();
                        case RandomForest:
                                return new RandomForest();
                        case RandomCommittee:
                                return new RandomCommittee();
                        case RandomTree:
                                return new RandomTree();
                        case ZeroR:
                                return new ZeroR();
                        case Stacking:
                                return new Stacking();
                        case AdaBoostM1:
                                return new AdaBoostM1();
                        case Bagging:
                                return new Bagging();
                        case ComplementNaiveBayes:
                                return new ComplementNaiveBayes();
                        case IB1:
                                return new IB1();
                        case J48:
                                return new J48();
                        case KStar:
                                return new KStar();
                        case LMT:
                                return new LMT();
                        case MultiScheme:
                                return new MultiScheme();
                        case NaiveBayes:
                                return new NaiveBayes();
                        case NaiveBayesMultinomial:
                                return new NaiveBayesMultinomial();
                        case OneR:
                                return new OneR();
                        case PART:
                                return new PART();
                        case RandomSubSpace:
                                return new RandomSubSpace();
                        case REPTree:
                                return new REPTree();
                        case SimpleLogistic:
                                return new SimpleLogistic();
                        case SMO:
                                return new SMO();
                        case LinearRegression:
                                return new LinearRegression();
                        default:
                                life = 0;
                                return null;
                }

        }

        private void getWekaDataset(Range range) {
                try {

                        FastVector attributes = new FastVector();

                        for (int i = 0; i < rowList.size(); i++) {
                                Attribute weight = new Attribute("weight" + i);
                                attributes.addElement(weight);
                        }

                        Attribute type = new Attribute("class");
                        attributes.addElement(type);

                        //Creates the dataset
                        this.training = new Instances("Dataset", attributes, 0);
                        this.test = new Instances("test Dataset", attributes, 0);

                        for (int i = 0; i < this.dataset.getNumberCols(); i++) {
                                if (!range.contains(i)) {
                                        String sampleName = dataset.getAllColumnNames().elementAt(i);

                                        double[] values = new double[training.numAttributes()];
                                        int cont = 0;
                                        for (PeakListRow row : rowList) {
                                                values[cont++] = (Double) row.getPeak(sampleName);
                                        }
                                        values[cont] = Double.valueOf(this.dataset.getSampleType(sampleName));

                                        Instance inst = new SparseInstance(1.0, values);
                                        training.add(inst);
                                } else {
                                        String sampleName = dataset.getAllColumnNames().elementAt(i);

                                        double[] values = new double[test.numAttributes()];
                                        int cont = 0;
                                        for (PeakListRow row : rowList) {
                                                values[cont++] = (Double) row.getPeak(sampleName);
                                        }
                                        values[cont] = Double.valueOf(this.dataset.getSampleType(sampleName));

                                        Instance inst = new SparseInstance(1.0, values);
                                        test.add(inst);

                                }
                        }

                        training.setClass(type);
                        test.setClass(type);


                } catch (Exception ex) {
                        Logger.getLogger(Bug.class.getName()).log(Level.SEVERE, null, ex);

                }

        }

        public double getTestError() {
                try {
                        Evaluation evalC = new Evaluation(training);
                       // evalC.crossValidateModel(classifier, training, 10, new Random(1));
                         evalC.evaluateModel(classifier, test);
                        double CVError = evalC.correlationCoefficient();
                        return CVError;
                } catch (Exception ex) {
                        ex.printStackTrace();
                        return -1.0;
                }
        }

        void setNewRange(Range newRange) {
                this.range = newRange;
        }

        public boolean isSameBug(Bug bug) {
                if (bug.getRows().size() != this.rowList.size()) {
                        return false;
                }
                for (PeakListRow val : bug.getRows()) {
                        if (!this.rowList.contains(val)) {
                                return false;
                        }
                }
                if (bug.getClassifierType().equals(this.classifierType)) {
                        return true;
                } else {
                        return false;
                }
        }
}
