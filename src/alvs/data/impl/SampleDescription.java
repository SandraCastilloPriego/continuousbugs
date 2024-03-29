/*
 * Copyright 2007-2010 
 * This file is part of ALVS.
 *
 * ALVS is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * ALVS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ALVS; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package alvs.data.impl;

import java.util.Hashtable;

/**
 * Stores the parameters values for one sample.
 * 
 * @author scsandra
 */
public class SampleDescription {

        Hashtable<String, String> parameters;

        
        public SampleDescription() {
                parameters = new Hashtable<String, String>();
        }

        /**
         * Adds the value of the parameter.
         *
         * @param parameterName  Name of the parameter
         * @param parameterValue Value of the parameter
         */
        public void addParameter(String parameterName, String parameterValue) {
                if (parameterName != null && parameterValue != null) {
                        parameters.put(parameterName, parameterValue);
                }
        }

        /**
         * Deletes the parameter
         *
         * @param parameterName Name of the parameter
         */
        public void deleteParameter(String parameterName) {
                if (parameters.containsKey(parameterName)) {
                        parameters.remove(parameterName);
                }
        }

        /**
         * Returns the parameter value.
         *
         * @param parameterName Name of the parameter
         * @return Parameter Value
         */
        public String getParameter(String parameterName) {
                if (parameterName != null && parameters.containsKey(parameterName)) {
                        return parameters.get(parameterName);
                } else {
                        return null;
                }
        }
}
