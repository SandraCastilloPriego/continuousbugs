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
package alvs.taskcontrol;

/**
 * @author Taken from MZmine2
 * http://mzmine.sourceforge.net/
 */
public enum TaskStatus {

        /**
         * WAITING - task is waiting for available thread
         *
         * PROCESSING - task is running
         *
         * FINISHED - task finished succesfully
         *
         * CANCELED - task was canceled by user
         *
         * ERROR - task finished with error, error message can be obtained by
         * getErrorMessage()
         *
         */
        WAITING, PROCESSING, FINISHED, CANCELED, ERROR
}
