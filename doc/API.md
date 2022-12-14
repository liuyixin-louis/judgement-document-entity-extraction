## 金额提取模块总接口

**地址**

www.lyx6178.cn:18080/exact_jine/

**方法**

POST

**接口说明**

**Header**

```json
{'Content-Type':  'application/json'}
```

**输入字段**

| 字段名 | 含义         | 备注   |
| ------ | ------------ | ------ |
| text   | 裁判文书全文 | 纯文本 |

**输出字段**

| 字段名              | 含义         | 备注                                                       |
| ------------------- | ------------ | ---------------------------------------------------------- |
| msg                 | 状态码       | 两种状态：0表示失败，需要规范请求的格式；1表示后台调用成功 |
| content             | 内容         | 根据处理后的各个金额字段，失败时为空字典。                 |
| 保证人-保证金额     |              | 详细介绍见场景说明书                                       |
| 利息截止时间        |              | 详细介绍见场景说明书                                       |
| 抵押物-最高抵押金额 |              | 详细介绍见场景说明书                                       |
| 金额实体            | 10类金额实体 | 详细介绍见场景说明书                                       |

**中英文对照**

保证人  guarantor
最高保证金额  maximum_guaranteed_amount
利息截止时间 deadline
抵押物 mortgage
贷款本金 debt_amount
利息(余额) interest
抵押物最高抵押金额 maximum_mortgage_amount
金额实体 amount
其他金额 other_amount
合计利息 total_interest
复利余额 compound_interest_balance
本息合计 sum_insterest_debt
本金余额 principal_balance
罚息余额 payment_balance



## 示例

### 成功请求

**请求体**：

```json
{"text":"广东省广州市越秀区人民法院\n民 事 判 决 书\n（2015）穗越法金民初字第2406号\n原告：上海浦东发展银行股份有限公司广州分行，住所地广州市天河区珠江新城珠江西路12号无限极大厦10楼。\n负责人：李荣军，行长。\n委托诉讼代理人：方介明，该分行工作人员。\n委托诉讼代理人：陈特炎，该分行工作人员。\n被告：广东沃帮投资有限公司，住所地广州市越秀区下塘西路87号首层自编之四。\n法定代表人：方广彬。\n被告：广东永安贸易有限公司，住所地广州市越秀区下塘西路89号109房间。\n法定代表人：王振豪。\n两被告共同委托诉讼代理人：方广柏，该公司工作人员。\n被告：方广彬，女，1964年6月22日出生，汉族，身份证住址广州市天河区。\n原告上海浦东发展银行股份有限公司广州分行与被告广东沃帮投资有限公司（以下简称沃帮公司）、广东永安贸易有限公司（以下简称永安公司）、方广彬金融借款合同纠纷一案，本院于2015年7月10日立案后，依法适用普通程序，公开开庭进行了审理。原告的诉讼代理人方介明到庭参加诉讼，被告永安公司、沃帮公司、方广彬经传票传唤无正当理由拒不到庭参加诉讼，本院依法作缺席审理。本案现已审理终结。\n原告向本院提出诉讼请求：1、被告沃帮公司偿还原告借款本金20000000元及相应的利息、罚息和复利（利息、罚息及复利的计算日期自产生时起计至欠款清偿时止，截至2015年6月20日所产生的利息和罚息为372088.89元）；2、被告永安公司对上述债务承担连带清偿责任；3、被告方广彬对上述债务承担连带清偿责任；4、原告对被告方广彬已抵押的房地产享有优先受偿权；5、本案的诉讼费用、保全费用由上述被告承担。事实和理由：2015年1月7日，上海浦东发展银行广州盘福支行（以下简称盘福支行）与被告沃帮公司签订《融资额度协议》，双方约定原告浦发银行盘福支行向被告沃帮公司提供20000000元整最高融资额度，额度使用期限为1年，即自2015年1月7日至2016年1月5日，融资品种为流动资金贷款。\n2014年8月7日，盘福支行与被告永安公司、方广彬分别签订《最高额保证合同》，约定被告永安公司、方广彬为原告在2015年1月7日至2016年1月5日期间内，为被告沃帮公司向原告提供连带保证担保。2013年5月21日，盘福支行与被告方广彬签订《最高额抵押合同》，被告方广彬以其名下的位于广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，位于广州市天河区华景北路84号3105房，位于增城市××山庄郁金香路××房产，为盘福支行的涉案债权提供抵押担保，并办理了有效的抵押登记手续。\n2015年1月27日，盘福支行和被告沃帮公司签订《流动资金借款合同》，约定原告提供20000000元的借款给被告沃帮公司，借款期限为一年。随后，盘福支行如约发放贷款给被告沃帮公司。被告沃帮公司未按期付息，构成违约。\n被告永安公司及沃帮公司在诉讼中辩称：对贷款本金20000000元予以确认，但不确认利息和罚息。\n经审理查明：2015年1月7日，盘福支行与被告沃帮公司签订《额度融资协议》，约定：盘福支行为被告沃帮公司提供额度为20000000的融资额度，额度循环方式为不可循环使用，适用融资品种为流动资金贷款，额度使用期限为2015年1月7日至2016年1月5日。同日，盘福支行与被告永安公司、方广彬分别签订《最高额保证合同》，约定：被告永安公司、方广彬为盘福支行在自2015年1月7日至2016年1月5日止期间与被告沃帮公司办理各类融资业务所发生的债权以及双方约定的在前债权（如有），在最高债权额分别为22230000元、22330000元内提供连带责任保证；保证范围除主债权之外，还包括由此产生的利息（包括利息、罚息和复利）、违约金、损害赔偿金、手续费等；保证期间为自债权合同债务履行期届满之日起至该债权合同约定的债务履行期届满之日后两年止。\n2013年5月21日，盘福支行与被告方广彬签订《最高额抵押合同》，约定：被告方广彬为盘福支行在自2013年5月21日至2015年5月21日止的期间内与被告沃帮公司办理各类融资业务所发生的债权，以及双方约定的在先债权（如有），在最高债权余额22230000元内提供抵押担保；抵押财产为被告方广彬名下的位于广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，位于广州市天河区华景北路84号3105房，位于增城市××山庄郁金香路××号的房产。上述抵押房产在相关管理部门办理了抵押登记，抵押权人均为盘福支行，最高债权额为22230000元。\n2015年1月27日，盘福支行（贷款人）与被告沃帮公司（借款人）签订《流动资金借款合同》，约定：盘福支行为被告沃帮公司提供短期流动资金贷款，借款金额为20000000元，具体用途为购买金属铝锭；借款期限为2015年1月27日至2016年1月10日；贷款利率为固定利率，合同项下的每笔贷款发放时按发放日贷款人公布的12个月的浦发银行贷款基础利率+173BPS计算；逾期罚息利率按计收罚息日适用的贷款执行利率加收30%执行；贷款结息方式为按季结息，结息日为每季末月的二十日等；贷款人对借款人不能按时支付的利息（包括正常利息、逾期罚息、挪用罚息），自逾期之日起，按本合同约定的逾期罚息利率按实际逾期天数计收复利。\n同日，盘福支行依约向被告沃帮公司发放贷款20000000元。被告沃帮公司取得贷款后未能依约还款，截至2015年6月20日，尚欠盘福支行借款本金20000000元，利息372088.89元。\n另查明，盘福支行为原告附属非独立核算分支机构，不具有独立法人资格。原告向中国银行业监督管理委员会广东监管局作出的《上海浦东发展银行广州分行关于下辖盘福支行终止营业的请示》中载明盘福支行在获取监管局同意之日起将终止营业，其现有对公及个人业务移交分行营业部继续办理等。中国银行监督管理委员会广东监管局于2015年9月25日作出的粤银监复【2015】442号《关于上海浦东发展银行广州盘福支行终止营业的批复》载明，同意上海浦东发展银行广州盘福支行终止营业，上海浦东发展银行广州分行，即本案原告，应妥善处理网点终止营业后的清理及善后工作等。根据广州市工商行政管理局于2016年8月25日出具的《企业核准注销登记通知书》记载，上海浦东发展银行股份有限公司广州盘福支行经审查，核准注销登记。\n本院认为：盘福支行原是具有贷款资格的金融机构，盘福支行与被告沃帮公司签订的《额度融资协议》、《流动资金借款合同》，盘福支行与被告永安公司、方广彬签订的《最高额保证合同》以及盘福支行与被告方广彬签订的《最高额抵押合同》是各方当事人在自愿、平等、协商一致的基础上形成的合意，合同内容没有违反法律、行政法规的强制性规定，合法有效，并对缔约当事人产生约束力，各方均应恪守执行。盘福支行根据《流动资金借款合同》的约定向被告沃帮公司发放了贷款20000000元，被告沃帮公司在收到贷款之后未能按照合同约定还本付息，构成严重违约，损害了盘福支行的合法债权。由于盘福支行已依法办理注销登记手续，原告系盘福支行的上级分行，且作为盘福支行对公及个人业务的接收分行，要求被告沃帮公司偿还尚欠借款本息的请求合法有据，本院予以支持。\n被告永安公司、方广彬与盘福支行签订《最高额保证合同》，自愿分别在22230000元、2233000元额度内对被告沃帮公司的上述主债务承担连带责任保证，故被告永安公司、方广彬依法应对被告沃帮公司的上述债务承担连带清偿责任。被告永安公司、方广彬承担保证责任后，有权按照《中华人民共和国担保法》第三十一条的规定向被告沃帮公司追偿。\n被告方广彬公司与盘福支行签订《最高额抵押合同》，自愿将其名下所有的广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，广州市天河区华景北路84号3105房，增城市××山庄郁金香路××号的房产抵押给盘福支行，在22230000元额度内为被告沃帮公司的借款提供抵押担保，并办理了抵押登记手续，原告的抵押权成立。在被告沃帮公司不履行还款义务时，原告有权在22230000元额度内，从依法处分抵押房产所得价款中优先受偿。\n综上所述，依照《中华人民共和国合同法》第八条、第六十条第一款、第二百零五条、第二百零六条、第二百零七条，《中华人民共和国物权法》第一百七十三条、第一百七十六条、第一百七十九条、第一百八十七条、第一百九十五条、第一百九十八条，《中华人民共和国担保法》第六条、第十八条、第二十一条、第三十一条、第三十三条之规定，判决如下：\n一、在本判决发生法律效力之日起十日内，被告广东沃帮投资有限公司向原告上海浦东发展银行广州分行偿还借款本金20000000元及利息（包含单利、罚息及复利，暂计至2015年6月20日止的单利、复利，合计为372088.89元；自2015年6月21日起至2016年1月10日止的单利、复利按照《流动资金借款合同》的约定计算；自2016年1月11日起至实际清偿之日止的罚息按照《流动资金借款合同》的约定计算）。\n二、被告广东永安贸易有限公司在22230000元的最高限额内对被告广东沃帮投资有限公司上述债务承担连带清偿责任；被告广东永安贸易有限公司承担保证责任后，有权依照《中华人民共和国担保法》第三十一条的规定向被告广东沃帮投资有限公司追偿。\n三、被告方广彬在22330000元的最高限额内对被告广东沃帮投资有限公司上述债务承担连带清偿责任；被告方广彬承担保证责任后，有权依照《中华人民共和国担保法》第三十一条的规定向被告广东沃帮投资有限公司追偿。\n四、被告广东沃帮投资有限公司不履行上述债务时，原告上海浦东发展银行广州分行有权以依法处分被告方广彬名下的广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，广州市天河区华景北路84号3105房，增城市新塘镇紫云山庄郁金香路9号房产所得的价款在22230000元额度内优先受偿。\n如果未按本判决指定的期间履行给付金钱义务，应当按照《中华人民共和国民事诉讼法》第二百五十三条之规定，加倍支付迟延履行期间的债务利息。\n案件受理费143660元，由被告广东沃帮投资有限公司、广东永安贸易有限公司、方广彬连带负担。\n如不服本判决，可在本判决书送达之日起十五日内向本院递交上诉状，并按对方当事人的人数提出副本，上诉于广东省广州市中级人民法院。\n审\u3000判\u3000长\u3000\u3000李文君\n人民陪审员\u3000\u3000林\u3000静\n人民陪审员\u3000\u3000徐燕琼\n二〇一六年九月二十九日\n书\u3000记\u3000员\u3000\u3000郭绮娜\n"}
```



**响应体：**

```json
{
    "content": {
        "amount": {
            "compound_interest_balance": [],
            "debt_amount": [
                [
                    20000000.0,
                    "元"
                ]
            ],
            "interest": [],
            "maximum_guaranteed_amount": [
                [
                    22230000.0,
                    "元"
                ],
                [
                    22330000.0,
                    "元"
                ]
            ],
            "maximum_mortgage_amount": [
                [
                    22230000.0,
                    "元"
                ]
            ],
            "other_amount": [],
            "payment_balance": [],
            "principal_balance": [
                [
                    20000000.0,
                    "元"
                ]
            ],
            "sum_insterest_debt": [],
            "total_interest": [
                [
                    372088.89,
                    "元"
                ]
            ]
        },
        "deadline": [
            "2015年6月20日"
        ],
        "guarantor-maximum_guaranteed_amount": [
            {
                "guarantor": [
                    "广东永安贸易有限公司",
                    "广东沃帮投资有限公司"
                ],
                "maximum_guaranteed_amount": [
                    22230000.0,
                    "元"
                ]
            },
            {
                "guarantor": [
                    "方广彬",
                    "广东沃帮投资有限公司"
                ],
                "maximum_guaranteed_amount": [
                    22330000.0,
                    "元"
                ]
            }
        ],
        "mortgage-maximum_mortgage_amount": [
            {
                "maximum_mortgage_amount": [
                    22230000.0,
                    "元"
                ],
                "mortgage": [
                    "广东沃帮投资有限公司不履行上述债务时，原告上海浦东发展银行广州分行有权以依法处分被告方广彬名下",
                    "广州市越秀区下塘西路89号",
                    "广州市天河区华景北路84号",
                    "增城市新塘镇紫云山庄郁金香路9号"
                ]
            }
        ]
    },
    "msg": 1
}
```

### 失败请求

**请求**

```json
{
    
}
```

**返回**

```json
{
    "content": {
        "amount": {
            "compound_interest_balance": [],
            "debt_amount": [],
            "interest": [],
            "maximum_guaranteed_amount": [],
            "maximum_mortgage_amount": [],
            "other_amount": [],
            "payment_balance": [],
            "principal_balance": [],
            "sum_insterest_debt": [],
            "total_interest": []
        },
        "deadline": [],
        "guarantor-maximum_guaranteed_amount": [],
        "mortgage-maximum_mortgage_amount": []
    },
    "msg": 0
}
```




## 提取抵押物地址-抵押金额关系对模块接口

**地址**

www.lyx6178.cn:18080/exact_mortgage/

**方法**

POST

**接口说明**

**Header**

```json
{'Content-Type':  'application/json'}
```

**输入字段**

| 字段名 | 含义         | 备注   |
| ------ | ------------ | ------ |
| text   | 裁判文书全文 | 纯文本 |

**输出字段**

| 字段名              | 含义         | 备注                                                       |
| ------------------- | ------------ | ---------------------------------------------------------- |
| msg                 | 状态码       | 两种状态：0表示失败，需要规范请求的格式；1表示后台调用成功 |
| content             | 内容         | 根据处理后数据                 |
| mortgage-maximum_mortgage_amount     |              | 抵押物-抵押金额对应关系                                       |
| maximum_mortgage_amount        |              | 抵押物金额                                       |
| mortgage |              | 抵押物                                       |

**中英文对照**

保证人  guarantor
最高保证金额  maximum_guaranteed_amount
利息截止时间 deadline
抵押物 mortgage
贷款本金 debt_amount
利息(余额) interest
抵押物最高抵押金额 maximum_mortgage_amount
金额实体 amount
其他金额 other_amount
合计利息 total_interest
复利余额 compound_interest_balance
本息合计 sum_insterest_debt
本金余额 principal_balance
罚息余额 payment_balance



## 示例

### 成功请求

**请求体**：

```json
{"text":"广东省广州市越秀区人民法院\n民 事 判 决 书\n（2015）穗越法金民初字第2406号\n原告：上海浦东发展银行股份有限公司广州分行，住所地广州市天河区珠江新城珠江西路12号无限极大厦10楼。\n负责人：李荣军，行长。\n委托诉讼代理人：方介明，该分行工作人员。\n委托诉讼代理人：陈特炎，该分行工作人员。\n被告：广东沃帮投资有限公司，住所地广州市越秀区下塘西路87号首层自编之四。\n法定代表人：方广彬。\n被告：广东永安贸易有限公司，住所地广州市越秀区下塘西路89号109房间。\n法定代表人：王振豪。\n两被告共同委托诉讼代理人：方广柏，该公司工作人员。\n被告：方广彬，女，1964年6月22日出生，汉族，身份证住址广州市天河区。\n原告上海浦东发展银行股份有限公司广州分行与被告广东沃帮投资有限公司（以下简称沃帮公司）、广东永安贸易有限公司（以下简称永安公司）、方广彬金融借款合同纠纷一案，本院于2015年7月10日立案后，依法适用普通程序，公开开庭进行了审理。原告的诉讼代理人方介明到庭参加诉讼，被告永安公司、沃帮公司、方广彬经传票传唤无正当理由拒不到庭参加诉讼，本院依法作缺席审理。本案现已审理终结。\n原告向本院提出诉讼请求：1、被告沃帮公司偿还原告借款本金20000000元及相应的利息、罚息和复利（利息、罚息及复利的计算日期自产生时起计至欠款清偿时止，截至2015年6月20日所产生的利息和罚息为372088.89元）；2、被告永安公司对上述债务承担连带清偿责任；3、被告方广彬对上述债务承担连带清偿责任；4、原告对被告方广彬已抵押的房地产享有优先受偿权；5、本案的诉讼费用、保全费用由上述被告承担。事实和理由：2015年1月7日，上海浦东发展银行广州盘福支行（以下简称盘福支行）与被告沃帮公司签订《融资额度协议》，双方约定原告浦发银行盘福支行向被告沃帮公司提供20000000元整最高融资额度，额度使用期限为1年，即自2015年1月7日至2016年1月5日，融资品种为流动资金贷款。\n2014年8月7日，盘福支行与被告永安公司、方广彬分别签订《最高额保证合同》，约定被告永安公司、方广彬为原告在2015年1月7日至2016年1月5日期间内，为被告沃帮公司向原告提供连带保证担保。2013年5月21日，盘福支行与被告方广彬签订《最高额抵押合同》，被告方广彬以其名下的位于广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，位于广州市天河区华景北路84号3105房，位于增城市××山庄郁金香路××房产，为盘福支行的涉案债权提供抵押担保，并办理了有效的抵押登记手续。\n2015年1月27日，盘福支行和被告沃帮公司签订《流动资金借款合同》，约定原告提供20000000元的借款给被告沃帮公司，借款期限为一年。随后，盘福支行如约发放贷款给被告沃帮公司。被告沃帮公司未按期付息，构成违约。\n被告永安公司及沃帮公司在诉讼中辩称：对贷款本金20000000元予以确认，但不确认利息和罚息。\n经审理查明：2015年1月7日，盘福支行与被告沃帮公司签订《额度融资协议》，约定：盘福支行为被告沃帮公司提供额度为20000000的融资额度，额度循环方式为不可循环使用，适用融资品种为流动资金贷款，额度使用期限为2015年1月7日至2016年1月5日。同日，盘福支行与被告永安公司、方广彬分别签订《最高额保证合同》，约定：被告永安公司、方广彬为盘福支行在自2015年1月7日至2016年1月5日止期间与被告沃帮公司办理各类融资业务所发生的债权以及双方约定的在前债权（如有），在最高债权额分别为22230000元、22330000元内提供连带责任保证；保证范围除主债权之外，还包括由此产生的利息（包括利息、罚息和复利）、违约金、损害赔偿金、手续费等；保证期间为自债权合同债务履行期届满之日起至该债权合同约定的债务履行期届满之日后两年止。\n2013年5月21日，盘福支行与被告方广彬签订《最高额抵押合同》，约定：被告方广彬为盘福支行在自2013年5月21日至2015年5月21日止的期间内与被告沃帮公司办理各类融资业务所发生的债权，以及双方约定的在先债权（如有），在最高债权余额22230000元内提供抵押担保；抵押财产为被告方广彬名下的位于广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，位于广州市天河区华景北路84号3105房，位于增城市××山庄郁金香路××号的房产。上述抵押房产在相关管理部门办理了抵押登记，抵押权人均为盘福支行，最高债权额为22230000元。\n2015年1月27日，盘福支行（贷款人）与被告沃帮公司（借款人）签订《流动资金借款合同》，约定：盘福支行为被告沃帮公司提供短期流动资金贷款，借款金额为20000000元，具体用途为购买金属铝锭；借款期限为2015年1月27日至2016年1月10日；贷款利率为固定利率，合同项下的每笔贷款发放时按发放日贷款人公布的12个月的浦发银行贷款基础利率+173BPS计算；逾期罚息利率按计收罚息日适用的贷款执行利率加收30%执行；贷款结息方式为按季结息，结息日为每季末月的二十日等；贷款人对借款人不能按时支付的利息（包括正常利息、逾期罚息、挪用罚息），自逾期之日起，按本合同约定的逾期罚息利率按实际逾期天数计收复利。\n同日，盘福支行依约向被告沃帮公司发放贷款20000000元。被告沃帮公司取得贷款后未能依约还款，截至2015年6月20日，尚欠盘福支行借款本金20000000元，利息372088.89元。\n另查明，盘福支行为原告附属非独立核算分支机构，不具有独立法人资格。原告向中国银行业监督管理委员会广东监管局作出的《上海浦东发展银行广州分行关于下辖盘福支行终止营业的请示》中载明盘福支行在获取监管局同意之日起将终止营业，其现有对公及个人业务移交分行营业部继续办理等。中国银行监督管理委员会广东监管局于2015年9月25日作出的粤银监复【2015】442号《关于上海浦东发展银行广州盘福支行终止营业的批复》载明，同意上海浦东发展银行广州盘福支行终止营业，上海浦东发展银行广州分行，即本案原告，应妥善处理网点终止营业后的清理及善后工作等。根据广州市工商行政管理局于2016年8月25日出具的《企业核准注销登记通知书》记载，上海浦东发展银行股份有限公司广州盘福支行经审查，核准注销登记。\n本院认为：盘福支行原是具有贷款资格的金融机构，盘福支行与被告沃帮公司签订的《额度融资协议》、《流动资金借款合同》，盘福支行与被告永安公司、方广彬签订的《最高额保证合同》以及盘福支行与被告方广彬签订的《最高额抵押合同》是各方当事人在自愿、平等、协商一致的基础上形成的合意，合同内容没有违反法律、行政法规的强制性规定，合法有效，并对缔约当事人产生约束力，各方均应恪守执行。盘福支行根据《流动资金借款合同》的约定向被告沃帮公司发放了贷款20000000元，被告沃帮公司在收到贷款之后未能按照合同约定还本付息，构成严重违约，损害了盘福支行的合法债权。由于盘福支行已依法办理注销登记手续，原告系盘福支行的上级分行，且作为盘福支行对公及个人业务的接收分行，要求被告沃帮公司偿还尚欠借款本息的请求合法有据，本院予以支持。\n被告永安公司、方广彬与盘福支行签订《最高额保证合同》，自愿分别在22230000元、2233000元额度内对被告沃帮公司的上述主债务承担连带责任保证，故被告永安公司、方广彬依法应对被告沃帮公司的上述债务承担连带清偿责任。被告永安公司、方广彬承担保证责任后，有权按照《中华人民共和国担保法》第三十一条的规定向被告沃帮公司追偿。\n被告方广彬公司与盘福支行签订《最高额抵押合同》，自愿将其名下所有的广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，广州市天河区华景北路84号3105房，增城市××山庄郁金香路××号的房产抵押给盘福支行，在22230000元额度内为被告沃帮公司的借款提供抵押担保，并办理了抵押登记手续，原告的抵押权成立。在被告沃帮公司不履行还款义务时，原告有权在22230000元额度内，从依法处分抵押房产所得价款中优先受偿。\n综上所述，依照《中华人民共和国合同法》第八条、第六十条第一款、第二百零五条、第二百零六条、第二百零七条，《中华人民共和国物权法》第一百七十三条、第一百七十六条、第一百七十九条、第一百八十七条、第一百九十五条、第一百九十八条，《中华人民共和国担保法》第六条、第十八条、第二十一条、第三十一条、第三十三条之规定，判决如下：\n一、在本判决发生法律效力之日起十日内，被告广东沃帮投资有限公司向原告上海浦东发展银行广州分行偿还借款本金20000000元及利息（包含单利、罚息及复利，暂计至2015年6月20日止的单利、复利，合计为372088.89元；自2015年6月21日起至2016年1月10日止的单利、复利按照《流动资金借款合同》的约定计算；自2016年1月11日起至实际清偿之日止的罚息按照《流动资金借款合同》的约定计算）。\n二、被告广东永安贸易有限公司在22230000元的最高限额内对被告广东沃帮投资有限公司上述债务承担连带清偿责任；被告广东永安贸易有限公司承担保证责任后，有权依照《中华人民共和国担保法》第三十一条的规定向被告广东沃帮投资有限公司追偿。\n三、被告方广彬在22330000元的最高限额内对被告广东沃帮投资有限公司上述债务承担连带清偿责任；被告方广彬承担保证责任后，有权依照《中华人民共和国担保法》第三十一条的规定向被告广东沃帮投资有限公司追偿。\n四、被告广东沃帮投资有限公司不履行上述债务时，原告上海浦东发展银行广州分行有权以依法处分被告方广彬名下的广州市越秀区下塘西路89号109房、1010房、209之一房、209之二号铺，广州市天河区华景北路84号3105房，增城市新塘镇紫云山庄郁金香路9号房产所得的价款在22230000元额度内优先受偿。\n如果未按本判决指定的期间履行给付金钱义务，应当按照《中华人民共和国民事诉讼法》第二百五十三条之规定，加倍支付迟延履行期间的债务利息。\n案件受理费143660元，由被告广东沃帮投资有限公司、广东永安贸易有限公司、方广彬连带负担。\n如不服本判决，可在本判决书送达之日起十五日内向本院递交上诉状，并按对方当事人的人数提出副本，上诉于广东省广州市中级人民法院。\n审\u3000判\u3000长\u3000\u3000李文君\n人民陪审员\u3000\u3000林\u3000静\n人民陪审员\u3000\u3000徐燕琼\n二〇一六年九月二十九日\n书\u3000记\u3000员\u3000\u3000郭绮娜\n"}
```



**响应体：**

```json
{
    "content": {
        "mortgage-maximum_mortgage_amount": [
            {
                "maximum_mortgage_amount": [
                    22230000.0,
                    "元"
                ],
                "mortgage": [
                    "广东沃帮投资有限公司不履行上述债务时，原告上海浦东发展银行广州分行有权以依法处分被告方广彬名下",
                    "广州市越秀区下塘西路89号",
                    "广州市天河区华景北路84号",
                    "增城市新塘镇紫云山庄郁金香路9号"
                ]
            }
        ]
    },
    "msg": 1
}
```

### 失败请求

**请求**

```json
{
    "content": {
        "mortgage-maximum_mortgage_amount": []
    },
    "msg": 0
}
```

**返回**

```json
{
    "content": {
        "mortgage-maximum_mortgage_amount": []
    },
    "msg": 0
}
```

