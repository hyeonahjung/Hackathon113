// 
//                           SimuLTE
// 
// This file is part of a software released under the license included in file
// "license.pdf". This license can be also found at http://www.ltesimulator.com/
// The above file and the present reference are part of the software itself, 
// and cannot be removed from it.
//
package lte.simulations.a_testworld_02;

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

network Highway2
{
    parameters:
        double playgroundSizeX @unit(m); // x size of the area the nodes are in (in meters)
        double playgroundSizeY @unit(m); // y size of the area the nodes are in (in meters)
        double playgroundSizeZ @unit(m); // z size of the area the nodes are in (in meters)
        @display("bgb=10000,4730.641");

    submodules:

        routingRecorder: RoutingTableRecorder {
            @display("p=853.3125,238.65001;is=s");
        }
        configurator: IPv4NetworkConfigurator {
            @display("p=380.17502,240.0375");
            config = xmldoc("demo.xml");
        }

        //# Veins manager module
        veinsManager: VeinsInetManager {
            @display("p=149.85,238.65001;is=s");
        }

        //# LTE modules
        channelControl: LteChannelControl {
            @display("p=1082.25,238.65001;is=s");
        }
        binder: LteBinder {
            @display("p=625.7625,240.0375;is=s");
        }
        server: StandardHost {
            @display("p=7705.5806,3738.9949;is=n;i=device/server");
        }
        router: Router {
            @display("p=5624.7485,3722.7383;i=device/smallrouter");
        }
        pgw: PgwStandardSimplified {
            nodeType = "PGW";
            @display("p=3235.0432,3462.6343;is=l");
        }
        eNodeB1: eNodeB {
            @display("p=4000,50;is=vl");
        }
        eNodeB2: eNodeB {
            @display("p=6000,50;is=vl");
        }
        droneB1: droneNodeB {
            @display("p=5000,50;is=vl");
        }
    connections allowunconnected:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> pgw.filterGate;
        pgw.pppg++ <--> Eth10G <--> eNodeB1.ppp;
        pgw.pppg++ <--> Eth10G <--> eNodeB2.ppp;
        pgw.pppg++ <--> Eth10G <--> droneB1.ppp;

        //# X2 connections
        eNodeB1.x2++ <--> Eth10G <--> droneB1.x2++;
        droneB1.x2++ <--> Eth10G <--> eNodeB2.x2++;
        eNodeB1.x2++ <--> Eth10G <--> eNodeB2.x2++;
}

