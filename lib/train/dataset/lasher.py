import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class LasHeR(BaseVideoDataset):
    """ LasHeR dataset for RGB-T tracking.

    Publication:
        LasHeR: A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/abs/2104.13202

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the lasher training data.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'test'.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasher_dir if root is None else root
        super().__init__('LasHeR_add', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self, split):
        if split == 'train':
            # with open(os.path.join(self.root, '..', 'testingsetList.txt')) as f:
            #     dir_list = list(csv.reader(f))
            dir_list = ['2ndblkboy1_quezhen', '2ndrunningboy', 'abreastinnerboy', 'ajiandan_blkdog', 'ajiandan_boyleft2right', 'ajiandan_catwhite', 'basketballatright', 'basketballup', 'bike2', 'bikeafterwhitecar', 'bikeboy128', 'blkbagontheleftgirl', 'blkboy', 'blkboycoming', 'blkboygoleft', 'blkboylefttheredbagboy', 'blkboywillstand', 'blkboywithbluebag', 'blkboywithglasses', 'blkboywithwhitebackpack', 'blkgirlfat_quezhen', 'blkgirlfromcolumn1_quezhen', 'blkteacher`shead', 'bluegirlcoming', 'bluegirlriding', 'bowblkboy1-quezhen', 'boy', 'boy1227', 'boyaftertworunboys', 'boyatbluegirlleft', 'boybackpack', 'boybehindtrees', 'boybehindtrees2', 'boycoming', 'boyindarkwithgirl', 'boyinsnowfield2', 'boyinsnowfield4', 'boyinsnowfield_inf_white', 'boyleft', 'boyplayingphone', 'boyputtinghandup', 'boyrightrubbish', 'boyrightthelightbrown', 'boyrunning', 'boytakingcamera', 'boytoleft_inf_white', 'boyunderthecolumn', 'boywalkinginsnow', 'boywalkinginsnow3', 'car', 'carleaves', "comingboy'shead", 'easy_rightblkboywithgirl', 'easy_runninggirls', 'easy_whiterignt2left', 'farmanrightwhitesmallhouse', 'firstleftrunning', 'fogboyscoming1_quezhen_inf_heiying', 'girlatleft', 'girlruns2right', 'girltakingplate', 'girlwithredhat', 'greenboy', 'greenboyafterwhite', 'greyboysit1_quezhen', 'hatboy`shead', 'lastblkboy1_quezhen', 'left2ndboy', 'left2ndgreenboy', 'left3boycoming', 'left3rdgirlbesideswhitepants', 'left3rdrunwaygirlbesideswhitepants', 'left4thgirlwithwhitepants', 'leftblkboy', 'leftbrowngirlfat', 'leftlightsweaterboy', 'leftorangeboy', 'leftrunningboy', 'leftshortgirl', 'lightredboy', 'manaftercars', 'manatwhiteright', 'manbesideslight', 'manfarbesidespool', 'manleftsmallwhitehouse', 'manrun', 'manupstairs', 'midblkboyplayingphone', 'midboyplayingphone', 'midwhitegirl', 'moto2', 'nightboy', 'nightrightboy1', 'orangegirl', 'pinkgirl', 'redgirl', 'redgirlafterwhitecar', 'redgirlsits', 'redlittleboy', 'right2ndblkboy', 'right4thboy', 'rightbiggreenboy', 'rightblkboy', 'rightblkboy2', 'rightblkboybesidesred', 'rightblkgirl', 'rightboy479', 'rightboyatwindow', 'rightboybesidesredcar', 'rightboystand', 'rightboywithbluebackpack', 'rightboywithwhite', 'rightestblkboy', 'rightestblkboy2', 'rightgreenboy', 'rightofth3boys', 'rightredboy', 'rightredboy1227', 'rightredboy954', 'rightwhitegirl', 'rightwhitegirlleftpink', 'runninggreenboyafterwhite', 'singleboywalking', 'sitleftboy', 'the2ndboyunderbasket', 'the4thboy', 'the4thboystandby', 'theleftboytakingball', 'theleftestrunningboy', 'trolleywith2boxes1_quezhen', 'waiterontheothersideofwindow', 'whiteboy1_quezhen', 'whitegirl', 'whitegirl1227', 'whitegirl2', 'whitegirl2right', 'whitegirlcoming', 'yellowgirlwithbowl', '2boys', '2boysup', '2girlgoleft', '2rdcarcome', '2rdtribike', '2up', '3bike2', '3blackboys', '3girl1', '3whitemen', '4boys2left', '4sisters', 'agirl', 'basketman', 'bigbus', 'bike150', 'bikeboygo', 'bikeboyleft', 'bikeinhand', 'blackbag', 'blackbaggirl', 'blackboy256', 'blackcar', 'blackinpeople', 'blackman2', 'blackpantsman', 'blackphoneboy', 'blackturnr', 'bluecar', 'boybehindbus', 'boyleft161', 'boyright', 'boysback', 'boyscome', 'boyturn', 'browncarturn', 'carclosedoor', 'carfarstart', 'cargirl2', 'carleaveturnleft', 'carstart', 'carstop', 'carturn', 'carturnleft', 'checkedshirt', 'comecar', 'cycleman', 'downmoto', 'easy_4women', 'easy_blackboy', 'etrike', 'girl2trees', 'girlbike', 'girlcoat', 'girlleaveboys', 'girlleft2right1', 'girlleft2right2', 'girlpickboy', 'girlumbrella', 'leftblackboy', 'man', 'manaftercar', 'mancrossroad', 'manglass1', 'manglass2', 'manphone', 'manwait1', 'man_head', 'man_with_black_clothes2', 'man_with_black_clothes3', 'minibus125', 'motobike', 'motocross', 'motoman', 'occludedmoto', 'rightbluegirl', 'threepeople', 'treeboy', 'twopeople', 'umbregirl', 'unbrellainbike', 'whiteblcakwoman', 'whiteboycome', 'whitecar', 'whiteman', 'whiteshirt', 'whitewoman', 'woman', 'womanopendoor', '2boysatblkcarend', '2boysbesidesblkcar', '2boyscome', '2boyscome245', '2girlinrain', '2girlsridebikes', '2gointrees', '2ndbus', '2ndcarcome', '2ndgirlmove', '2outdark', '2sisiters', '3rdboy', '4boysbesidesblkcar', 'ab_boyfromtrees', 'ab_girlrideintrees', 'ab_moto2north0', 'ab_motocometurn', 'basketball', 'basketboyblack', 'basketboywhite', 'bike2north', 'bikeblkbag', 'bikeblkturn', 'bikeboy173', 'bikeboycome', 'bikeboystrong', 'bikecoming', 'biked', 'bikefromnorth', 'bikefromnorth2', 'bikefromnorth257', 'bikeorange', 'biketonorth', 'biketurn', 'bikeumbrellacome', 'bikewithbag', 'blackaftertrees', 'blackbagbike', 'blackboypushbike', 'blackcar126', 'blackcar131', 'blackcarcome', 'blackcargo', 'blackcarturn175', 'blackcarturn183', 'blackmanleft', 'blackof4bikes', 'blackridebike', 'blackridebike2', 'blacktallman', 'blkbikefromnorth', 'blkboyatbike', 'blkboyback636', 'blkboyonleft', 'blkboywithblkbag', 'blkboywithumbrella', 'blkcar2north', 'blkcarcome', 'blkcarcome155', 'blkcarcomeinrain', 'blkcarfollowingwhite', 'blkcargo', 'blkcarinrain', 'blkgirlbike', 'blkmaninrain', 'blkmoto', 'blkmotocome', 'blueboy85', 'blueboybike', 'blueboycome', 'blueboywalking', 'bluegirl', 'bluelittletruck', 'bluemanatbike', 'blueumbrellagirl', 'boyalone', 'boybesidesblkcarrunning', 'boybesidescarwithouthat', 'boybetween2blkcar', 'boybikeblueumbrella', 'boybikewithbag', 'boyblackback', 'boycome', 'boydown', 'boyfromdark2', 'boygointrees', 'boyleave', 'boyouttrees', 'boyride2trees', 'boyrideoutandin', 'boyridesbike', 'boyrun', 'boyshead', 'boyshead2', 'boyshorts', 'boysitbike', 'boysumbrella', 'boysumbrella2', 'boysumbrella3', 'boytakebox', 'boytakepath', 'boytakesuicase', 'boyumbrella4', 'boywithshorts', 'boywithshorts2', 'bus', 'bus2', 'bus2north', 'car2north', 'car2north2', 'car2north3', 'carbesidesmoto', 'carcomeonlight2', 'carfromnorth', 'carfromnorth2', 'carlight', 'carout', 'carstart2east', 'darkgiratbike', 'dogfollowinggirl', 'doginrain', 'dogouttrees', 'drillmaster', 'e-tribike', 'e-tricycle', 'farredcar', 'farwhiteboy', 'fatmancome', 'folddenumbrellainhand', 'girlaftertree', 'girlalone', 'girlbesidesboy', 'girlbike156', 'girlbikeinlight', 'girlblack', 'girlfoldumbrella', 'girlgoleft', 'girlridesbike', 'girltakebag', 'girltakemoto', 'goaftrtrees', 'gonemoto_ab', 'greenboywithgirl', 'greengirls', 'guardatbike_ab', 'huggirl', 'jeepblack', 'jeepleave', 'leftblkboy648', 'leftboy', 'leftgirl', 'leftof2girls', 'lightcarcome', 'lightcarfromnorth', 'lightcarstart', 'lightcarstop', 'lonelyman', 'man2startmoto', 'manaftercar114', 'manaftertrees', 'manatmoto', 'manbikecoming', 'mancarstart', 'manfromcar', 'manfromcar302', 'maninfrontofbus', 'manopendoor', 'manrun250', 'manstarttreecar', 'mengointrees', 'midboy', 'midgirl', 'minibus152', 'minibuscome', 'moto2north', 'moto2north1', 'moto2north2', 'moto2trees', 'moto2trees2', 'moto78', 'motobesidescar', 'motocome', 'motocome122', 'motocomenight', 'motofromdark', 'motoinrain', 'motolight', 'motoprecede', 'motoslow', 'motosmall', 'mototake2boys', 'mototurn', 'mototurn102', 'mototurn134', 'mototurnleft', 'mototurnright', 'motowithblack', 'motowithgood', 'motowithtopcoming', 'nikeatbike', 'oldwoman', 'pinkbikeboy', 'raincarstop', 'rainyboyaftertrees', 'rainysuitcase', 'rainywhitecar', 'redboywithblkumbrella', 'redcar', 'redcarturn', 'redmotocome', 'redshirtman', 'redup', 'redwhitegirl', 'rightblkboy188', 'rightgirl', 'rightgirlbike', 'schoolbus', 'silvercarcome', 'sisterswithbags', 'stripeman', 'stronggirl', 'stubesideswhitecar', 'suitcase', 'take-out-motocoming', 'takeoutman', 'takeoutman953', 'takeoutmanleave', 'takeoutmoto', 'tallboyblack', 'tallwhiteboy', 'the4thwhiteboy', 'trashtruck', 'truck', 'truckcoming', 'truckk', 'truckwhite', 'umbellaatnight', 'umbrellabikegirl', 'umbrellainblack', 'umbrellainred', 'umbrellainyellowhand', 'whiteaftertree', 'whiteaftertrees', 'whiteatright', 'whiteboy', 'whiteboy395', 'whiteboyatbike', 'whiteboybike', 'whiteboycome598', 'whiteboyphone', 'whiteboyright', 'whiteboywait', 'whiteboywithbag', 'whitecar70', 'whitecarafterbike', 'whitecarcome', 'whitecarcome192', 'whitecarcomes', 'whitecarcoming', 'whitecarfromnorth', 'whitecargo', 'whitecarinrain', 'whitecarleave', 'whitecarleave198', 'whitecarstart', 'whitecarstart126', 'whitecarturn', 'whitecarturn2', 'whitecarturn85', 'whitecarturn137', 'whitecarturn178', 'whitecarturnl', 'whitecarturnl2', 'whitedown', 'whitegirl209', 'whiteskirtgirl', 'whitesuvcome', 'whiteTboy', 'womanaroundcar', 'womanongrass', 'yellowgirl', 'yellowumbrellagirl', '2ndbikecoming', 'ab_bikeboycoming', 'ab_minibusstops', 'ab_motoinrain', 'ab_mototurn', 'ab_redboyatbike', 'ab_shorthairgirlbike', 'bike2trees86', 'bikecome', 'bikecoming176', 'bikefromwest', 'bikeout', 'biketurndark', 'biketurnleft', 'biketurnleft2', 'blackboy186', 'blackcarback', 'blackcarcoming', 'blkbikecomes', 'blkboy198', 'blkboybike', 'blkcarcome115', 'blkcarinrain107', 'blkcarstart', 'blkman2trees', 'blkmototurn', 'blkridesbike', 'blkskirtwoman', 'blktakeoutmoto', 'bluebike', 'blueboyopenbike', 'bluemanof3', 'bluemoto', 'bluetruck', 'boycomingwithumbrella', 'boymototakesgirl', 'boyridesbesidesgirl', 'boywithblkbackpack', 'boywithumbrella', 'browncar2east', 'browncar2north', 'bus2north111', 'camonflageatbike', 'carstart189', 'carstarts', 'carturncome', 'carturnleft109', 'comebike', 'darkredcarturn', 'dogunderthelamp', 'farwhitecarturn', 'girlinfrontofcars', 'girlintrees', 'girlplayingphone', 'girlshakeinrain', 'girltakingmoto', 'girlthroughtrees', 'girlturnbike', 'girlwithblkbag', 'girlwithumbrella', 'greenboy438', 'guardman', 'leftgirlafterlamppost', 'leftgirlunderthelamp', 'leftwhitebike', 'lightmotocoming', 'manafetrtrees', 'mantoground', 'manwalkincars', 'manwithyellowumbrella', 'meituanbike', 'meituanbike2', 'midblkbike', 'minibusback', 'moto2ground', 'moto2north101', 'moto2west', 'motocome2left', 'motocomeinrain', 'motocometurn', 'motocoming', 'motocominginlight', 'motoinrain56', 'motolightturnright', 'motostraught2east', 'mototake2boys123', 'mototaking2boys', 'mototakinggirl', 'mototurntous', 'motowithtop', 'nearmangotoD', 'nightmototurn', 'openningumbrella', 'orangegirlwithumbrella', 'pinkgirl285', 'rainblackcarcome', 'raincarstop2', 'redbaginbike', 'redgirl2trees', 'redminirtruck', 'redumbrellagirlcome', 'rightboywitjbag', 'rightgirlatbike', 'rightgirlbikecome', 'rightgirlwithumbrella', 'rightof2boys', 'runningwhiteboy', 'shunfengtribike', 'skirtwoman', 'smallmoto', 'takeoutmoto521', 'takeoutmototurn', 'trimototurn', 'turnblkbike', 'whitebikebehind', 'whitebikebehind172', 'whitebikebehind2', 'whiteboyback', 'whitecar2west', 'whitecarback', 'whitecarstart183', 'whitecarturnright248', 'whitegirlatbike', 'whitegirlcrossingroad', 'whitegirlundertheumbrella', 'whitegirlwithumbrella', 'whitemancome', 'whiteminibus197', 'whitemoto', 'whitemotoout', 'whiteof2boys', 'whitesuvstop', 'womanstartbike', 'yellowatright', 'yellowcar', 'yellowtruck', '10crosswhite', '10phone_boy', '10rightblackboy', '10rightboy', '11righttwoboy', '11runone', '11runthree', '1boygo', '1handsth', '1phoneblue', '1rightboy', '1righttwogreen', '1whiteteacher', '2runfive', '2runfour', '2runone', '2runsix', '2runtwo', '2whitegirl', '4four', '4one', '4runeight', '4runone', '4thgirl', '4three', '4two', '5manwakeright', '5numone', '5one', '5runfour', '5runone', '5runthree', '5runtwo', '5two', '6walkgirl', '7one', '7rightblueboy', '7rightredboy', '7rightwhitegirl', '7runone', '7runthree', '7runtwo', '7two', '8lastone', '9handlowboy', '9hatboy', '9whitegirl', 'abeauty_1202', 'aboyleft_1202', 'aboy_1202', 'ab_bolster', 'ab_catescapes', 'ab_hyalinepaperatground', 'ab_leftfoam', 'ab_leftmirrordancing', 'ab_pingpongball', 'ab_pingpongball3', 'ab_rightcupcoming_infwhite_quezhen', 'ab_righthandfoamboard', 'ab_rightmirror', 'actor_1202', 'agirl1_1202', 'agirl_1202', 'battlerightblack', 'blackbetweengreenandorange', 'blackdownball', 'blackdresswithwhitefar', 'blackman_0115', 'blackthree_1227', 'blklittlebag', 'blkumbrella', 'Blue_in_line_1202', 'bolster', 'bolster_infwhite', 'bookatfloor', 'boy2_0115', 'boy2_1227', 'boy_0109', 'boy_0115', 'boy_1227', 'cameraman_1202', 'camera_1202', 'catbrown', 'dogforward', 'dotat43', 'downwhite_1227', 'elector_0115', 'elector_1227', 'exercisebook', 'fallenbikeitself', 'foamboardatlefthand', 'folderatlefthand', 'folderinrighthand', 'foundsecondpeople_0109', 'frontmirror', 'girlafterglassdoor2', 'girlatwindow', 'girlback', 'girloutreading', 'girlrightcomein', 'greenfaceback', 'greenleftbackblack', 'greenleftthewhite', 'greenrightblack', 'higherthwartbottle_quezhen', 'hyalinepaperfrontclothes', 'left4thblkboyback', 'leftbottle', 'leftclosedexersicebook', 'leftcup', 'leftfallenchair_inf_white', 'lefthandfoamboard', 'lefthyalinepaper', 'leftmirror2', 'leftmirrorshining', 'leftredcup', 'leftthrowfoam', 'left_first_0109', 'left_two_0109', 'lovers_1227', 'lover_1202', 'lowerfoam2throw', 'man_0109', 'midpinkblkglasscup', 'mirroratleft', 'mirrorfront', 'nearestleftblack', 'openthisexersicebook', 'orange', 'othersideoftheriver_1227', 'outer2leftmirrorback', 'outerfoam', 'peoplefromright_0109', 'pickuptheyellowbook', 'pingpongball', 'pingpongball2', 'pingpongpad', 'pingpongpad2', 'redcupatleft', 'right2ndblkpantsboy', 'rightbackcup', 'rightbattle', 'rightbottle', 'rightboy_1227', 'rightexercisebookwillfly', 'rightgreen', 'righthand`sfoam', 'righthunchblack', 'rightmirrorbackwards', 'rightmirrorlikesky', 'rightmirrornotshining', 'rightof2cupsattached', 'rightredcup_quezhen', 'rightshiningmirror', 'rightstripeblack', 'rightumbrella_quezhen', 'rightwhite_1227', 'shotmaker', 'shotmaker2', 'swan2_0109', 'Take_an_umbrella_1202', 'thefirstexcersicebook', 'The_girl_back_at_the_lab_1202', 'The_girl_with_the_cup_1202', 'The_one_on_the_left_in_black_1202', 'twopeople_0109', 'twoperpson_1202', 'twoperson_1202', 'two_1227', 'wanderingly_1202', 'whitacatfrombush', 'whitebetweenblackandblue', 'whiteboy242', 'whitecat', 'whitecatjump', 'whitegirl2_0115', 'whitegirl_0115', 'whitewoman_1202', 'yellowexcesicebook', 'Aab_whitecarturn', 'Ablkboybike77', 'Abluemotoinrain', 'Aboydownbike', 'Acarlightcome', 'Agirlrideback', 'Ahercarstart', 'Amidredgirl', 'Amotoinrain150', 'Amotowithbluetop', 'AQbikeback', 'AQblkgirlbike', 'AQboywithumbrella415', 'AQgirlbiketurns', 'AQmanfromdarktrees', 'AQmidof3boys', 'AQmotomove', 'AQraincarturn2', 'AQrightofcomingmotos', 'AQtaxi', 'AQwhiteminibus', 'AQwhitetruck', 'Aredtopmoto', 'Athe3rdboybesidescar', 'Awhitecargo', 'Awhiteleftbike', 'Awoman2openthecardoor', '1rowleft2ndgirl', '1strow3rdboymid', '1strowleft3rdgirl', '1strowright1stboy', '1strowright2ndgirl', '1strowrightgirl', '2ndboyfarintheforest2right', 'backpackboyhead', 'basketballatboysrighthand', 'basketballbyNo_9boyplaying', 'basketballshooting', 'basketballshooting2', 'belowrightwhiteboy', 'belowyellow-gai', 'besom-ymm', 'besom2-gai', 'besom4', 'besom5-sq', 'besom6', 'blackruning', 'bord', 'bouheadupstream', 'boyalonecoming', 'boybesidesbar2putcup', 'boyfollowing', 'boyof2leaders', 'boyplayingphone366', 'boyshead509', 'boystandinglefttree', 'boyunderleftbar', 'boywithwhitebackpack', 'collegeofmaterial-gai', 'darkleftboy2left', 'elegirl', 'fardarkboyleftthe1stgirl', 'firstboythroughtrees', 'firstrightflagcoming', 'girl', 'girl2-gai', 'girloutqueuewithbackpack', 'girlshead', 'girlsheadwithhat', 'girlsquattingbesidesleftbar', 'girltakingblkumbrella', 'girl`sheadoncall', 'glassesboyhead', 'highright2ndboy', 'large2-gai', 'large3-gai', 'lastgirl-qzc', 'lastof4boys', 'lastrowrightboy', 'left11', 'left2flagfornews', 'left2ndgirl', 'left4throwboy', 'leftaloneboy-gai', 'leftbasketball', 'leftblkboyunderbasketballhoop', 'leftboy-gai', 'leftboyleftblkbackpack', 'leftbroom', 'leftconerbattle-gai', 'leftconergirl', 'leftdress-gai', 'leftdrillmasterstanding', 'leftdrillmasterstandsundertree', 'leftgirl1299', 'leftgirlat1row', 'leftgirlchecking', 'leftgirlchecking2', 'leftlastboy-sq', 'leftlastgirl-yxb', 'leftlastgirl2', 'leftmen-chong1', 'leftredflag-lsz', 'leftwhiteblack', 'left_leader', 'midboyblue', 'midflag-qzc', 'midtallboycoming', 'nearstrongboy', 'ninboy-gai', 'notebook-gai', 'redbackpackgirl', 'redbaggirlleftgreenbar', 'redgirl1497', 'right1stgirlin2ndqueue', 'right2nddrillmaster', 'right2ndfarboytakinglight2left', 'right2ndgirl', 'rightbhindbike-gai', 'rightblkgirlNo_11', 'rightblkgirlrunning', 'rightbottle-gai', 'rightbottle2-gai', 'rightboyleader', 'rightboywithbackpackandumbrella', 'rightboy`shead', 'rightdrillmasterunderthebar', 'rightfirstboy-ly', 'rightfirstgirl-gai', 'rightgirlplayingphone', 'rightholdball', 'rightholdball1096', 'rightof2boys953', 'rightof2cominggirls', 'rightofthe4girls', 'rightrunninglatterone', 'righttallholdball', 'righttallnine-gai', 'rightwhiteboy', 'runningwhiteboy249', 'schoolofeconomics-yxb', 'small2-gai', 'strongboy`head', 'tallboyNumber_9', 'toulan-ly', 'twoleft', 'twolinefirstone-gai', 'twopeopleelec-gai', 'tworightbehindboy-gai', 'whiteboy-gai', 'whiteboyup', 'whiteboy`head', 'whitehatgirl`sheadleftwhiteumbrella', 'whiteshoesleftbottle-gai']
        else:
            # with open(os.path.join(self.root, '..', 'trainingsetList.txt')) as f:
            #     dir_list = list(csv.reader(f))
            dir_list = ['10runone', '11leftboy', '11runtwo', '1blackteacher', '1boycoming', '1stcol4thboy', '1strowleftboyturning', '1strowrightdrillmaster', '1strowrightgirl3540', '2girl', '2girlup', '2runseven', '3bike1', '3men', '3pinkleft', '3rdfatboy', '3rdgrouplastboy', '3thmoto', '4men', '4thboywithwhite', '7rightorangegirl', 'AQgirlwalkinrain', 'AQtruck2north', 'ab_bikeoccluded', 'ab_blkskirtgirl', 'ab_bolstershaking', 'ab_girlchoosesbike', 'ab_girlcrossroad', 'ab_pingpongball2', 'ab_rightlowerredcup_quezhen', 'ab_whiteboywithbluebag', 'advancedredcup', 'baggirl', 'ballshootatthebasket3times', 'basketball849', 'basketballathand', 'basketboy', 'bawgirl', 'belowdarkgirl', 'besom3', 'bike', 'bike2left', 'bike2trees', 'bikeboy', 'bikeboyintodark', 'bikeboyright', 'bikeboyturn', 'bikeboyturntimes', 'bikeboywithumbrella', 'bikefromlight', 'bikegoindark', 'bikeinrain', 'biketurnright', 'blackboy', 'blackboyoncall', 'blackcarturn', 'blackdown', 'blackgirl', 'blkboy`shead', 'blkboyback', 'blkboybetweenredandwhite', 'blkboydown', 'blkboyhead', 'blkboylefttheNo_21', 'blkboystand', 'blkboytakesumbrella', 'blkcaratfrontbluebus', 'blkgirlumbrella', 'blkhairgirltakingblkbag', 'blkmoto2north', 'blkstandboy', 'blktribikecome', 'blueboy', 'blueboy421', 'bluebuscoming', 'bluegirlbiketurn', 'bottlebetweenboy`sfeet', 'boy2basketballground', 'boy2buildings', 'boy2trees', 'boy2treesfindbike', 'boy`headwithouthat', 'boy`sheadingreycol', 'boyaftertree', 'boyaroundtrees', 'boyatdoorturnright', 'boydownplatform', 'boyfromdark', 'boyinlight', 'boyinplatform', 'boyinsnowfield3', 'boyleftblkrunning2crowd', 'boylefttheNo_9boy', 'boyoncall', 'boyplayphone', 'boyride2path', 'boyruninsnow', 'boyscomeleft', 'boyshead9684', 'boyss', 'boytakingbasketballfollowing', 'boytakingplate2left', 'boyunder2baskets', 'boywaitgirl', 'boywalkinginsnow2', 'broom', 'carbehindtrees', 'carcomeonlight', 'carcomingfromlight', 'carcominginlight', 'carlight2', 'carlightcome2', 'caronlight', 'carturn117', 'carwillturn', 'catbrown2', 'catbrownback2bush', 'couple', 'darkcarturn', 'darkgirl', 'darkouterwhiteboy', 'darktreesboy', 'drillmaster1117', 'drillmasterfollowingatright', 'farfatboy', 'firstexercisebook', 'foamatgirl`srighthand', 'foldedfolderatlefthand', 'girl2left3man1', 'girl`sblkbag', 'girlafterglassdoor', 'girldownstairfromlight', 'girlfromlight_quezhen', 'girlinrain', 'girllongskirt', 'girlof2leaders', 'girlrightthewautress', 'girlunderthestreetlamp', 'guardunderthecolumn', 'hugboy', 'hyalinepaperfrontface', 'large', 'lastleftgirl', 'leftblkTboy', 'leftbottle2hang', 'leftboy2jointhe4', 'leftboyoutofthetroop', 'leftchair', 'lefterbike', 'leftexcersicebookyellow', 'leftfarboycomingpicktheball', "leftgirl'swhitebag", 'lefthyalinepaper2rgb', 'lefthyalinepaperfrontpants', 'leftmirror', 'leftmirrorlikesky', 'leftmirrorside', 'leftopenexersicebook', 'leftpingpongball', 'leftrushingboy', 'leftunderbasket', 'leftuphand', 'littelbabycryingforahug', 'lowerfoamboard', 'mandownstair', 'manfromtoilet', 'mangetsoff', 'manoncall', 'mansimiliar', 'mantostartcar', 'midblkgirl', 'midboyNo_9', 'middrillmaster', 'midgreyboyrunningcoming', 'midof3girls', 'midredboy', 'midrunboywithwhite', 'minibus', 'minibusgoes2left', 'moto', 'motocomeonlight', 'motogoesaloongS', 'mototaking2boys306', 'mototurneast', 'motowithbluetop', 'pingpingpad3', 'pinkwithblktopcup', 'raincarturn', 'rainycarcome_ab', 'redboygoright', 'redcarcominginlight', 'redetricycle', 'redmidboy', 'redroadlatboy', 'redtricycle', 'right2ndflagformath', 'right5thflag', 'rightbike', 'rightbike-gai', 'rightblkboy4386', 'rightblkboystand', 'rightblkfatboyleftwhite', 'rightbluewhite', 'rightbottlecomes', 'rightboy504', 'rightcameraman', 'rightcar-chongT', 'rightcomingstrongboy', 'rightdarksingleman', 'rightgirltakingcup', 'rightwaiter1_quezhen', 'runningcameragirl', 'shinybikeboy2left', 'shinycarcoming', 'shinycarcoming2', 'silvercarturn', 'small-gai', 'standblkboy', 'swan_0109', 'truckgonorth', 'turning1strowleft2ndboy', 'umbreboyoncall', 'umbrella', 'umbrellabyboy', 'umbrellawillbefold', 'umbrellawillopen', 'waitresscoming', 'whitebikebelow', 'whiteboyrightcoccergoal', 'whitecarcomeinrain', 'whitecarturn683', 'whitecarturnleft', 'whitecarturnright', 'whitefardown', 'whitefargirl', 'whitegirlinlight', 'whitegirltakingchopsticks', 'whiteofboys', 'whiteridingbike', 'whiterunningboy', 'whiteskirtgirlcomingfromgoal', 'whitesuvturn', 'womanback2car', 'yellowgirl118', 'yellowskirt']

        # dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "init.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.ByteTensor([1 for v in range(len(bbox))])
        visible_ratio = torch.ByteTensor([1 for v in range(len(bbox))])

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        vis_frame_names = sorted(os.listdir(os.path.join(seq_path, 'visible')))
        inf_frame_names = sorted(os.listdir(os.path.join(seq_path, 'infrared')))

        return os.path.join(seq_path, 'visible', vis_frame_names[frame_id]), os.path.join(seq_path, 'infrared', inf_frame_names[frame_id])

    def _get_frame(self, seq_path, frame_id):
        path = self._get_frame_path(seq_path, frame_id)
        return np.concatenate((self.image_loader(path[0]),self.image_loader(path[1])), 2)

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta
