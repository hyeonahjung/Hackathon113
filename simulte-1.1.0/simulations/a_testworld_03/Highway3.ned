// 
//                           SimuLTE
// 
// This file is part of a software released under the license included in file
// "license.pdf". This license can be also found at http://www.ltesimulator.com/
// The above file and the present reference are part of the software itself, 
// and cannot be removed from it.
//
package lte.simulations.a_testworld_03;

import inet.networklayer.configurator.ipv4.IPv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.AdhocHost;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import inet.node.ethernet.Eth10G;

import lte.world.radio.LteChannelControl;
import lte.epc.PgwStandardSimplified;
import lte.corenetwork.binder.LteBinder;
import lte.corenetwork.nodes.droneNodeB;
import lte.corenetwork.nodes.eNodeB;
import lte.corenetwork.nodes.Ue;
import lte.corenetwork.nodes.cars.Car;

import org.car2x.veins.subprojects.veins_inet.VeinsInetManager;

network Highway3
{
    parameters:
        double playgroundSizeX @unit(m); // x size of the area the nodes are in (in meters)
        double playgroundSizeY @unit(m); // y size of the area the nodes are in (in meters)
        double playgroundSizeZ @unit(m); // z size of the area the nodes are in (in meters)
        @display("bgb=10000,3000");

    submodules:

        routingRecorder: RoutingTableRecorder {
            @display("p=1000,2000;is=s");
        }
        configurator: IPv4NetworkConfigurator {
            @display("p=800,2000");
            config = xmldoc("demo.xml");
        }

        //# Veins manager module
        veinsManager: VeinsInetManager {
            @display("p=600,2000;is=s");
        }

        //# LTE modules
        channelControl: LteChannelControl {
            @display("p=200,2000;is=s");
        }
        binder: LteBinder {
            @display("p=400,2000;is=s");
        }
        server: StandardHost {
            @display("p=7000,2000;is=n;i=device/server");
        }
        router: Router {
            @display("p=6000,2000;i=device/smallrouter");
        }
        pgw: PgwStandardSimplified {
            nodeType = "PGW";
            @display("p=5000,2000;is=l");
        }
        //#A
        eNodeB1: eNodeB {
            @display("p=0,50;is=vl");
        }
        droneB1: droneNodeB {
            @display("p=1266,50;is=vl");
        }
        droneB2: droneNodeB {
            @display("p=2067,50;is=vl");
        }
        //#B       
        eNodeB2: eNodeB {
            @display("p=3333,50;is=vl");
        }
        droneB3: droneNodeB {
            @display("p=4599,50;is=vl");
        }
        droneB4: droneNodeB {
            @display("p=5400,50;is=vl");
        }
        //#C
        eNodeB3: eNodeB {
            @display("p=6667,50;is=vl");
        }
        droneB5: droneNodeB {
            @display("p=7933,50;is=vl");
        }
        droneB6: droneNodeB {
            @display("p=8734,50;is=vl");
        }
        //#D
        eNodeB4: eNodeB {
            @display("p=10000,50;is=vl");
        }
//        record: recordf { //record module: gilteun
//            @display("p=1263.005,2788.615;i=device/disk");
//        }
    connections allowunconnected:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> pgw.filterGate;
        //#A
        pgw.pppg++ <--> Eth10G <--> eNodeB1.ppp;
        pgw.pppg++ <--> Eth10G <--> droneB1.ppp;
        pgw.pppg++ <--> Eth10G <--> droneB2.ppp;
        //#B
        pgw.pppg++ <--> Eth10G <--> eNodeB2.ppp;
        pgw.pppg++ <--> Eth10G <--> droneB3.ppp;
        pgw.pppg++ <--> Eth10G <--> droneB4.ppp;
        //#C
        pgw.pppg++ <--> Eth10G <--> eNodeB3.ppp;
        pgw.pppg++ <--> Eth10G <--> droneB5.ppp;
        pgw.pppg++ <--> Eth10G <--> droneB6.ppp;
        //#D
        pgw.pppg++ <--> Eth10G <--> eNodeB4.ppp;


        //# X2 connections

        //#A
        eNodeB1.x2++ <--> Eth10G <--> droneB1.x2++;
        droneB1.x2++ <--> Eth10G <--> droneB2.x2++;
        droneB2.x2++ <--> Eth10G <--> eNodeB2.x2++;
        //#B
        eNodeB2.x2++ <--> Eth10G <--> droneB3.x2++;
        droneB3.x2++ <--> Eth10G <--> droneB4.x2++;
        droneB4.x2++ <--> Eth10G <--> eNodeB3.x2++;
        //#C
        eNodeB3.x2++ <--> Eth10G <--> droneB5.x2++;
        droneB5.x2++ <--> Eth10G <--> droneB6.x2++;
        droneB6.x2++ <--> Eth10G <--> eNodeB4.x2++;
        //#D
}

