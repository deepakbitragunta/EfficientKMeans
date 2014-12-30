
package weka.clusterers;

import java.io.*;
import java.util.*;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.experiment.Stats;
import weka.classifiers.rules.DecisionTable;

/**
 * Fast k means clustering class.
 * 
 * Valid options are:
 * 
 * 
 * -N <number of clusters> 
 * Specify the number of clusters to generate.

 * 
 * -S<seed>
 * Specify random number seed.

 */
public class FastKMeans extends SimpleKMeans {

    private final static int s_minNumSamplesPerNode = 60;
    
	private static int treeDepth = 1;
    
    
	/**
	 * identifies the nearest cluster for each instance in a group
	 * of instances, while concurrently updating each cluster's error
	 * 
	 * 
	 * @param data
	 * 				Instances to cluster
	 * 
	 * @param oldClusterAssignments
	 * 				Previous cluster assignments, used to determine
	 * 				whether or not to return null.  If this parameter
	 * 				is null then it is not used, and this function
	 * 				will always return a non-null value.
	 * @return
	 * 				null if the clusters have not changed, else an int[] 
	 * 				array containing cluster indices for each
	 * 				item in data
	 */
	protected int[] clusterProcessedInstances(Instances instances, 
			int[] oldClusterAssignments) {
		
//		System.out.println(" FastKMeans");

        int numDimensions = instances.numAttributes();
        int numClusters = m_ClusterCentroids.numInstances();
		
        double[] minValues = new double[numDimensions];
        double[] maxValues = new double[numDimensions];
        double[] linearSums = new double[numDimensions];
        populateStatistics(instances, numDimensions, 
                minValues, maxValues, linearSums);

        double[][] closestDist = new double[numClusters][numDimensions];
        double[][] furthestDist = new double[numClusters][numDimensions];
        for(int i = 0; i < numDimensions; i++) {
        	double[] clustersPosAlongDim = m_ClusterCentroids.attributeToDoubleArray(i);
        	for(int j = 0; j < numClusters; j++) {
        		double distToMin = Math.abs(clustersPosAlongDim[j] - minValues[i]);
        		double distToMax = Math.abs(clustersPosAlongDim[j] - maxValues[i]);
        		if(distToMin < distToMax) {
        			closestDist[j][i] = distToMin;
        			furthestDist[j][i] = distToMax;
        		} else {
        			closestDist[j][i] = distToMax;
        			furthestDist[j][i] = distToMin;
        		}
        	}
        }
        double[] minDists = new double[numClusters];
        double[] maxDists = new double[numClusters];
        for(int i = 0; i < numClusters; i++) {
        	double minAccum = 0;
        	double maxAccum = 0;
        	for(int j = 0; j < numDimensions; j++) {
        		minAccum += sqr(closestDist[i][j]);
        		maxAccum += sqr(furthestDist[i][j]);
        	}
        	minDists[i] = Math.sqrt(minAccum);
        	maxDists[i] = Math.sqrt(maxAccum);
        }
        
		boolean sameAsOld = (oldClusterAssignments == null ? false : true);
		treeDepth = 1;
		int[] newClusterAssignments = treeAssociate(instances, m_ClusterCentroids, 
				closestDist, furthestDist, minDists, maxDists);
		
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance toCluster = instances.instance(i);
			if(sameAsOld) {
				if (newClusterAssignments[i] != oldClusterAssignments[i]) {
					sameAsOld = false;
				}
			}
		}
		
		if(sameAsOld) {
			return null;
		} else {
			return newClusterAssignments;
		}
	}    
    
    public int[] treeAssociate(Instances data, Instances candidateClusters, 
    		double[][] closestDistPriors, double[][] furthestDistPriors, 
            double[] minDistPriors, double[] maxDistPriors) {
    	
//    	System.out.println("tree depth: " + treeDepth);
        
        int numSamples = data.numInstances();
        int numDimensions = data.numAttributes();
        
        if(candidateClusters.numAttributes() != numDimensions) {
        	throw new RuntimeException();
        }
        
        // populate statistics
        double[] minValues = new double[numDimensions];
        double[] maxValues = new double[numDimensions];
        double[] linearSums = new double[numDimensions];
        populateStatistics(data, numDimensions, 
                minValues, maxValues, linearSums);
        
        // chose the longest dimension
        double maxDimLengthValue = Double.MIN_VALUE;
        int maxDimLengthIndex = -1;
        for(int i = 0; i < numDimensions; i++) {
            double dimLength = maxValues[i] - minValues[i];
            if(dimLength > maxDimLengthValue){
                maxDimLengthValue = dimLength;
                maxDimLengthIndex = i;
            }
        }
        
        // we're partitioning along maxDimLengthIndex into 2 partitions
        double partitionPoint = ((maxValues[maxDimLengthIndex] - 
                minValues[maxDimLengthIndex]) / 2) + 
                minValues[maxDimLengthIndex];
        Instances partLT = new Instances(data, 0);
        Instances partGT = new Instances(data, 0);
        Instances[] partArr = new Instances[] { partLT, partGT };
        int[][] parentAnswerIndex = new int[2][data.numInstances()];
        int[] parentAnswerIndexPointer = new int[2];
        for(int i = 0; i < data.numInstances(); i++) {
            Instance dataItem = data.instance(i);
            if(dataItem.value(maxDimLengthIndex) < partitionPoint) {
                partLT.add(dataItem);
                parentAnswerIndex[0][parentAnswerIndexPointer[0]++] = i;
            } else {
                partGT.add(dataItem);
                parentAnswerIndex[1][parentAnswerIndexPointer[1]++] = i;
            }
        }
        
        int[] clusterAssignments = new int[data.numInstances()];
        for(int partition = 0; partition < 2; partition++) {
            int numClusters = candidateClusters.numInstances();
            double[][] furthestDist = new double[numClusters][numDimensions];
            double[] maxDist = new double[numClusters];
            double[][] closestDist = new double[numClusters][numDimensions];
            double[] minDist = new double[numClusters];
            double minCutDimVal = (partition == 0 ? minValues[maxDimLengthIndex]
                                                  : partitionPoint);
            double maxCutDimVal = (partition == 0 ? partitionPoint
                                                  : maxValues[maxDimLengthIndex]);
            double minMaxDist = Double.MAX_VALUE;
            
            // calculate min and max distances for each cluster, to the points
            //   in partArr[partition]...use parents information since only
            //   one dimension in the hyperrectangle has been changed
            for(int i = 0; i < numClusters; i++) {
                double cutDimVal = candidateClusters.instance(i).value(maxDimLengthIndex);

                System.arraycopy(furthestDistPriors[i], 0, furthestDist[i], 0, maxDimLengthIndex);
                System.arraycopy(closestDistPriors[i], 0, closestDist[i], 0, maxDimLengthIndex);
                
                if(Math.abs(minCutDimVal - cutDimVal) > 
                        Math.abs(maxCutDimVal - cutDimVal)) {
                    furthestDist[i][maxDimLengthIndex] = cutDimVal - minCutDimVal;
                    closestDist[i][maxDimLengthIndex] = maxCutDimVal - cutDimVal;
                } else {
                    furthestDist[i][maxDimLengthIndex] = maxCutDimVal - cutDimVal;
                    closestDist[i][maxDimLengthIndex] = cutDimVal - minCutDimVal;
                }
                
                if(numDimensions-maxDimLengthIndex > 1) {
                    System.arraycopy(furthestDistPriors[i], maxDimLengthIndex+1,
                            furthestDist[i], maxDimLengthIndex+1, numDimensions-maxDimLengthIndex-1);
                    System.arraycopy(closestDistPriors[i], maxDimLengthIndex+1,
                            closestDist[i], maxDimLengthIndex+1, numDimensions-maxDimLengthIndex-1);
                }
                
                maxDist[i] = Math.sqrt(sqr(maxDistPriors[i]) - 
                        sqr(furthestDistPriors[i][maxDimLengthIndex]) + 
                        sqr(furthestDist[i][maxDimLengthIndex]));
                
                // check invariant
                double maxDistCheck = 0;
                for(int j = 0; j < numDimensions; j++) {
                	maxDistCheck += sqr(furthestDist[i][j]);
                }
                maxDistCheck = Math.sqrt(maxDistCheck);
                if(Math.abs(maxDistCheck - maxDist[i]) > .001) {
                	throw new RuntimeException();
                }
                
                minDist[i] = Math.sqrt(sqr(minDistPriors[i]) - 
                        sqr(closestDistPriors[i][maxDimLengthIndex]) + 
                        sqr(closestDist[i][maxDimLengthIndex]));
                
                // check invariant
                double minDistCheck = 0;
                for(int j = 0; j < numDimensions; j++) {
                	minDistCheck += sqr(closestDist[i][j]);
                }
                minDistCheck = Math.sqrt(minDistCheck);
                if(Math.abs(minDistCheck - minDist[i]) > .000001) {
                	throw new RuntimeException();
                }
                
                if(maxDist[i] < minMaxDist) {
                    minMaxDist = maxDist[i];
                }
                
                if(maxDist[i] < minDist[i]) {
                	throw new RuntimeException();
                }
                
                if(Double.isNaN(maxDist[i]) || Double.isNaN(minDist[i])) {
                	throw new RuntimeException();
                }
            }
            
            //  clusters using minMaxDist
            Instances newCandidateClusters = new Instances(candidateClusters, 0);
            int[] newIndices = new int[numClusters];
            int newIndicesPointer = 0;
            for(int i = 0; i < numClusters; i++) {
                if(minDist[i] < minMaxDist) {
                    // this cluster is still eligible
                	newIndices[newIndicesPointer++] = i;
                    newCandidateClusters.add(candidateClusters.instance(i));
                }
            }
            
            if(newCandidateClusters.numInstances() == 0) {
            	throw new RuntimeException();
            }
            
            int[] childAssignments = null;
            if(newCandidateClusters.numInstances() == 1) {
            	
            	childAssignments = new int[partArr[partition].numInstances()];
            	for(int i = 0; i < childAssignments.length; i++) {
            		childAssignments[i] = 0;
            	}
            	
            } else if(partArr[partition].numInstances() < s_minNumSamplesPerNode) {
               
            	
            	childAssignments = bruteForceAssociate(partArr[partition], 
            			newCandidateClusters);
            	
            } else {
            	// construct children
            	
            	treeDepth++;
            	childAssignments = treeAssociate(partArr[partition],
            			newCandidateClusters, closestDist, furthestDist, minDist,
            			maxDist);
            	treeDepth--;
            }
            
        	for(int i = 0; i < childAssignments.length; i++) {
        		int parentDataInstanceIndex = parentAnswerIndex[partition][i];
        		int parentClusterIndex = newIndices[childAssignments[i]];
        		clusterAssignments[parentDataInstanceIndex] = parentClusterIndex;
        	}
        }
        
        return clusterAssignments;
    }
    
    private int[] bruteForceAssociate(Instances data, 
    		Instances candidateClusters) {
    	
    	int[] associations = new int[data.numInstances()];
    	
        for(int i = 0; i < data.numInstances(); i++) {
            Instance dataItem = data.instance(i);
            double minDist = Double.MAX_VALUE;
            int minDistIndex = -1;
            
            for(int j = 0; j < candidateClusters.numInstances(); j++) {
            	Instance cluster = candidateClusters.instance(j);

            	double dist = distance(dataItem,cluster);
            	
			if(dist < minDist) {
            		minDist = dist;
            		minDistIndex = j;
            	}
            }
            
            associations[i] = minDistIndex;
        }
    	
    	return associations;
    }
    
    /*
    private static double distance(Instance x, Instance y) {
    	// assume x and y are compatible
    	
    	double dist = 0;
    	double[] xArr = x.toDoubleArray();
    	double[] yArr = y.toDoubleArray();
    	for(int i = 0; i < xArr.length; i++) {
    		dist += sqr(xArr[i] - yArr[i]);
    	}
    	return Math.sqrt(dist);
    }*/
    
    private static void populateStatistics(Instances data, int numDimensions,
            double[] minValuesToPopulate, double[] maxValuesToPopulate,
            double[] linearSumToPopulate) {
        
        boolean dimIsEven = (numDimensions % 2 == 0);
        for(int i = 0; i < numDimensions; i++) {
            double[] allValues = data.attributeToDoubleArray(i);

            double maxValue = Double.MIN_VALUE;
            double minValue = Double.MAX_VALUE;
            
            if(!dimIsEven) {
                minValue = maxValue = linearSumToPopulate[i] = allValues[0];
            }
            
            for(int j = (dimIsEven ? 0 : 1); j < data.numInstances()-1; j+=2) {                
                // linearSum
                linearSumToPopulate[i] += allValues[j];
                linearSumToPopulate[i] += allValues[j+1];
                
                // min-max in pairs
                if(allValues[j] < allValues[j+1]) {
                    if(minValue > allValues[j]) {
                        minValue = allValues[j];
                    }
                    if(maxValue < allValues[j+1]) {
                        maxValue = allValues[j+1];
                    }
                } else {
                    if(minValue > allValues[j+1]) {
                        minValue = allValues[j+1];
                    }
                    if(maxValue < allValues[j]) {
                        maxValue = allValues[j];
                    }
                }
            }
            
            minValuesToPopulate[i] = minValue;
            maxValuesToPopulate[i] = maxValue;
        }
    }
    
    private static double sqr(double x) {
        return x*x;
    }

	/**
	 * Main method for testing this class.
	 * 
	 * @param argv
	 *            should contain the following arguments:
	 *            <p>
	 *            -t training file [-N number of clusters]
	 */
	public static void main(String[] argv) {
		try {
			System.out.println(ClusterEvaluation.evaluateClusterer(
					new FastKMeans(), argv));
		} catch (Exception e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
	}
}