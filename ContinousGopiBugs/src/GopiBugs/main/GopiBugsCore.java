/*
 * Copyright 2007-2010 
 * This file is part of GopiBugs.
 *
 * GopiBugs is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * GopiBugs is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * GopiBugs; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package GopiBugs.main;

import GopiBugs.data.StorableParameterSet;
import GopiBugs.data.ParameterSet;

import GopiBugs.desktop.Desktop;
import GopiBugs.desktop.impl.MainWindow;
import GopiBugs.desktop.impl.helpsystem.HelpImpl;
import GopiBugs.taskcontrol.TaskController;
import GopiBugs.util.NumberFormatter;
import GopiBugs.util.NumberFormatter.FormatterType;
import java.io.File;
import java.io.FileWriter;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.JOptionPane;



import org.dom4j.Document;
import org.dom4j.Element;
import org.dom4j.io.OutputFormat;
import org.dom4j.io.SAXReader;
import org.dom4j.io.XMLWriter;

/**
 * This interface represents GopiBugs core modules - I/O, task controller and GUI.
 *
 * @author Taken from MZmine2
 * http://mzmine.sourceforge.net/
 */
public abstract class GopiBugsCore {

    public static final File CONFIG_FILE = new File("conf/config.xml");
    // configuration XML structure
    public static final String PARAMETER_ELEMENT_NAME = "parameter";
    public static final String PARAMETERS_ELEMENT_NAME = "parameters";
    public static final String MODULES_ELEMENT_NAME = "modules";
    public static final String MODULE_ELEMENT_NAME = "module";
    public static final String CLASS_ATTRIBUTE_NAME = "class";
    public static final String NODES_ELEMENT_NAME = "nodes";
    public static final String LOCAL_ATTRIBUTE_NAME = "local";
    public static final String DESKTOP_ELEMENT_NAME = "desktop";
    public static final String PREFERENCES_ELEMENT_NAME = "preferences";
    private static Logger logger = Logger.getLogger(GopiBugsCore.class.getName());
    protected static TaskController taskController;
    protected static Desktop desktop;
    protected static GopiBugsModule[] initializedModules;
    protected static GopiBugsPreferences preferences;
    protected static HelpImpl help;

    public static String getGopiBugsVersion() {
        return " 1.01";
    }

    /**
     * Returns a reference to local task controller.
     *
     * @return TaskController reference
     */
    public static TaskController getTaskController() {
        return taskController;
    }

    /**
     * Returns a reference to Desktop. May return null on GopiBugs nodes with no
     * GUI.
     *
     * @return Desktop reference or null
     */
    public static Desktop getDesktop() {
        return desktop;
    }

  
    /**
     * Returns an array of all initialized GopiBugs modules
     *
     * @return Array of all initialized GopiBugs modules
     */
    public static GopiBugsModule[] getAllModules() {
        return initializedModules;
    }

    /**
     * Saves configuration and exits the application.
     *
     */
    public static void exitGopiBugs() {
        saveConfiguration(CONFIG_FILE);
        // If we have GUI, ask if use really wants to quit
        if (desktop != null) {
            int selectedValue = JOptionPane.showInternalConfirmDialog(desktop.getMainFrame().getContentPane(),
                    "Are you sure you want to exit GopiBugs?", "Exiting...",
                    JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);

            if (selectedValue != JOptionPane.YES_OPTION) {
                return;
            }
            desktop.getMainFrame().dispose();
        }

        System.exit(0);

    }

    public static void saveConfiguration(File file) {

        try {

            // load current configuration from XML
            SAXReader reader = new SAXReader();
            Document configuration = reader.read(CONFIG_FILE);
            Element configRoot = configuration.getRootElement();

            /*Element preferencesConfigElement = configRoot.element(PREFERENCES_ELEMENT_NAME);
            if (preferencesConfigElement == null) {
            preferencesConfigElement = configRoot.addElement(PREFERENCES_ELEMENT_NAME);
            }*
            preferencesConfigElement.clearContent();
            try {
            preferences.exportValuesToXML(preferencesConfigElement);
            } catch (Exception e) {
            logger.log(Level.SEVERE, "Could not save preferences", e);
            }

             */
            // save desktop configuration
            StorableParameterSet desktopParameters = ((MainWindow) desktop).getParameterSet();
            Element desktopConfigElement = configRoot.element(DESKTOP_ELEMENT_NAME);
            if (desktopConfigElement == null) {
                desktopConfigElement = configRoot.addElement(DESKTOP_ELEMENT_NAME);
            }
            desktopConfigElement.clearContent();
            try {
                desktopParameters.exportValuesToXML(desktopConfigElement);
            } catch (Exception e) {
                logger.log(Level.SEVERE,
                        "Could not save desktop configuration", e);
            }

            // traverse modules
            for (GopiBugsModule module : getAllModules()) {

                ParameterSet currentParameters = module.getParameterSet();
                if ((currentParameters == null) || (!(currentParameters instanceof StorableParameterSet))) {
                    continue;
                }

                String className = module.getClass().getName();
                String xpathLocation = "//configuration/modules/module[@class='" + className + "']";
                Element moduleElement = (Element) configuration.selectSingleNode(xpathLocation);
                if (moduleElement != null) {

                    Element parametersElement = moduleElement.element(PARAMETERS_ELEMENT_NAME);
                    if (parametersElement == null) {
                        parametersElement = moduleElement.addElement(PARAMETERS_ELEMENT_NAME);
                    } else {
                        parametersElement.clearContent();
                    }

                    try {
                        ((StorableParameterSet) currentParameters).exportValuesToXML(parametersElement);
                    } catch (Exception e) {
                        logger.log(Level.SEVERE,
                                "Could not save configuration of module " + module, e);
                    }
                }

            }

            // write the config file
            OutputFormat format = OutputFormat.createPrettyPrint();
            XMLWriter writer = new XMLWriter(new FileWriter(file), format);
            writer.write(configuration);
            writer.close();

            logger.info("Saved configuration to file " + file);

        } catch (Exception e) {
            logger.log(Level.SEVERE, "Could not update configuration file " + file, e);
        }

    }

    public static void loadConfiguration(File file) {

        try {
            SAXReader reader = new SAXReader();
            Document configuration = reader.read(file);
            Element configRoot = configuration.getRootElement();

            logger.finest("Loading desktop configuration");

            /*Element preferencesConfigElement = configRoot.element(PREFERENCES_ELEMENT_NAME);
            if (preferencesConfigElement != null) {
            preferences.importValuesFromXML(preferencesConfigElement);
            }
             */
            StorableParameterSet desktopParameters = (StorableParameterSet) desktop.getParameterSet();
            Element desktopConfigElement = configRoot.element(DESKTOP_ELEMENT_NAME);
            if (desktopConfigElement != null) {
                desktopParameters.importValuesFromXML(desktopConfigElement);
            }

            logger.finest("Loading modules configuration");

            for (GopiBugsModule module : getAllModules()) {
                String className = module.getClass().getName();
                String xpathLocation = "//configuration/modules/module[@class='" + className + "']";

                Element moduleElement = (Element) configuration.selectSingleNode(xpathLocation);
                if (moduleElement == null) {
                    continue;
                }

                Element parametersElement = moduleElement.element(PARAMETERS_ELEMENT_NAME);

                if (parametersElement != null) {
                    ParameterSet moduleParameters = module.getParameterSet();
                    if ((moduleParameters != null) && (moduleParameters instanceof StorableParameterSet)) {
                        ((StorableParameterSet) moduleParameters).importValuesFromXML(parametersElement);
                    }
                }

            }

            logger.info("Loaded configuration from file " + file);

        } catch (Exception e) {
            logger.log(Level.SEVERE, "Could not parse configuration file " + file, e);
        }

    }

    public static HelpImpl getHelpImpl() {
        return help;
    }

    public static GopiBugsPreferences getPreferences() {
        return preferences;
    }

    public static void setDesktop(MainWindow mainWindow) {
        desktop = mainWindow;
    }

    // Number formatting functions
    public static NumberFormatter getIntensityFormat() {
        return preferences.getIntensityFormat();
    }

    public static NumberFormatter getMZFormat() {
        if (preferences == null || preferences.getMZFormat() == null) {
            return new NumberFormatter(FormatterType.NUMBER, "0.000");
        }
        return preferences.getMZFormat();
    }

    public static NumberFormatter getRTFormat() {
        if (preferences == null || preferences.getRTFormat() == null) {
            return new NumberFormatter(FormatterType.TIME, "m:ss");
        }
        return preferences.getRTFormat();
    }
}
